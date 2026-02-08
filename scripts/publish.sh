#!/bin/bash
# LaTeX and document publishing workflow script

set -e

PROJECT_ROOT="$(dirname "$0")/.."

print_usage() {
    echo "Usage: $0 [COMMAND] [PROJECT/FILE] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  compile <project/file>  Compile LaTeX document (project/paper or project/file.tex)"
    echo "  quarto <project/file>   Render Quarto document (project/file.qmd)"
    echo "  clean [project]         Clean build artifacts (optionally for specific project)"
    echo "  watch <project/file>    Watch and auto-compile LaTeX file"
    echo "  setup                   Install LaTeX dependencies (requires system LaTeX)"
    echo "  notebook <notebook>     Convert notebook to PDF via LaTeX"
    echo ""
    echo "Examples:"
    echo "  $0 compile slt/quasi-singular-models/paper/slt-quasi-singular"
    echo "  $0 compile slt/quasi-singular-models  # compiles all .tex files in project"
    echo "  $0 quarto slt/quasi-singular-models/paper/notebook_to_paper"
    echo "  $0 notebook projects/slt/quasi-singular-models/notebooks/fit_dataset.ipynb"
    echo "  $0 clean slt/quasi-singular-models"
}

compile_latex() {
    local file="$1"
    local project_dir=""
    local tex_file=""

    # Parse project/file format
    if [[ "$file" == *"/"* ]]; then
        project_dir="projects/$(dirname "$file")"
        tex_file="$(basename "$file")"
    else
        echo "❌ Please specify project/file format (e.g., slt-quasi-singular/paper)"
        exit 1
    fi

    # Add .tex extension if not present
    if [[ "$tex_file" != *.tex ]]; then
        tex_file="${tex_file}.tex"
    fi

    if [[ ! -f "$project_dir/${tex_file}" ]]; then
        echo "❌ File $project_dir/${tex_file} not found"
        exit 1
    fi

    echo "🔄 Compiling ${tex_file} in project $(dirname "$file")..."
    cd "$project_dir"

    # Run pdflatex twice for references
    pdflatex -interaction=nonstopmode "${tex_file}" > /dev/null

    # Run bibtex if bibliography exists
    local base_name="${tex_file%.tex}"
    if [[ -f "${base_name}.aux" ]] && grep -q "bibdata" "${base_name}.aux" 2>/dev/null; then
        echo "📚 Processing bibliography..."
        bibtex "${base_name}" > /dev/null 2>&1 || true
        pdflatex -interaction=nonstopmode "${tex_file}" > /dev/null
    fi

    pdflatex -interaction=nonstopmode "${tex_file}" > /dev/null

    if [[ -f "${base_name}.pdf" ]]; then
        echo "✅ PDF generated: $project_dir/${base_name}.pdf"
    else
        echo "❌ PDF generation failed"
        exit 1
    fi
}

render_quarto() {
    local file="$1"
    local project_dir=""
    local qmd_file=""

    # Parse project/file format
    if [[ "$file" == *"/"* ]]; then
        project_dir="projects/$(dirname "$file")"
        qmd_file="$(basename "$file")"
    else
        echo "❌ Please specify project/file format (e.g., slt-quasi-singular/analysis)"
        exit 1
    fi

    # Add .qmd extension if not present
    if [[ "$qmd_file" != *.qmd ]]; then
        qmd_file="${qmd_file}.qmd"
    fi

    if [[ ! -f "$project_dir/${qmd_file}" ]]; then
        echo "❌ File $project_dir/${qmd_file} not found"
        exit 1
    fi

    echo "🔄 Rendering ${qmd_file} in project $(dirname "$file")..."
    cd "$PROJECT_ROOT"

    if command -v quarto >/dev/null 2>&1; then
        quarto render "$project_dir/${qmd_file}"
        echo "✅ Quarto document rendered: $project_dir/${qmd_file%.qmd}.pdf"
    else
        echo "⚠️  Quarto not found, using Python nbconvert..."
        uv run jupyter nbconvert --to pdf "$project_dir/${qmd_file}" 2>/dev/null || {
            echo "❌ Install Quarto CLI for best results: https://quarto.org/docs/get-started/"
            exit 1
        }
    fi
}

convert_notebook() {
    local notebook="$1"
    if [[ ! -f "$notebook" ]]; then
        echo "❌ Notebook $notebook not found"
        exit 1
    fi

    echo "🔄 Converting notebook to PDF..."
    local basename=$(basename "$notebook" .ipynb)
    local project_dir=$(dirname "$notebook")
    local output="${project_dir}/${basename}_output.pdf"

    if command -v quarto >/dev/null 2>&1; then
        quarto render "$notebook" --to pdf --output-dir "$project_dir"
        echo "✅ Notebook converted: $output"
    else
        uv run jupyter nbconvert --to pdf "$notebook" --output-dir "$project_dir"
        echo "✅ Notebook converted: $output"
    fi
}

clean_artifacts() {
    local project="$1"

    if [[ -n "$project" ]]; then
        echo "🧹 Cleaning LaTeX build artifacts in project $project..."
        cd "projects/$project"
        rm -f *.aux *.log *.bbl *.blg *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
        echo "✅ Build artifacts cleaned in project $project"
    else
        echo "🧹 Cleaning LaTeX build artifacts in all projects..."
        find projects/ -name "*.aux" -o -name "*.log" -o -name "*.bbl" -o -name "*.blg" \
                      -o -name "*.out" -o -name "*.toc" -o -name "*.fls" \
                      -o -name "*.fdb_latexmk" -o -name "*.synctex.gz" \
                      | xargs rm -f
        echo "✅ Build artifacts cleaned in all projects"
    fi
}

watch_latex() {
    local file="$1"
    local project_dir=""
    local tex_file=""

    # Parse project/file format
    if [[ "$file" == *"/"* ]]; then
        project_dir="projects/$(dirname "$file")"
        tex_file="$(basename "$file")"
    else
        echo "❌ Please specify project/file format (e.g., slt-quasi-singular/paper)"
        exit 1
    fi

    # Add .tex extension if not present
    if [[ "$tex_file" != *.tex ]]; then
        tex_file="${tex_file}.tex"
    fi

    echo "👀 Watching ${tex_file} in project $(dirname "$file") for changes (Ctrl+C to stop)..."

    if command -v fswatch >/dev/null 2>&1; then
        fswatch -o "$project_dir/${tex_file}" | while read f; do
            echo "🔄 File changed, recompiling..."
            compile_latex "$file"
        done
    elif command -v inotifywait >/dev/null 2>&1; then
        while inotifywait -e modify "$project_dir/${tex_file}" >/dev/null 2>&1; do
            echo "🔄 File changed, recompiling..."
            compile_latex "$file"
        done
    else
        echo "❌ Install fswatch (macOS) or inotify-tools (Linux) for watch functionality"
        exit 1
    fi
}

setup_latex() {
    echo "🔧 Setting up LaTeX environment..."

    # Install Python dependencies
    echo "📦 Installing Python publishing tools..."
    cd "$PROJECT_ROOT"
    uv sync --group publishing

    # Check for system LaTeX
    if command -v pdflatex >/dev/null 2>&1; then
        echo "✅ LaTeX found: $(pdflatex --version | head -n1)"
    else
        echo "⚠️  LaTeX not found. Install options:"
        echo "  • macOS: brew install --cask mactex-no-gui"
        echo "  • Ubuntu: sudo apt-get install texlive-latex-extra texlive-fonts-recommended"
        echo "  • Arch: sudo pacman -S texlive-core texlive-latexextra"
        echo "  • Or use Tectonic: cargo install tectonic"
    fi

    # Check for Quarto
    if command -v quarto >/dev/null 2>&1; then
        echo "✅ Quarto found: $(quarto --version)"
    else
        echo "⚠️  Install Quarto CLI for best notebook→PDF conversion:"
        echo "    https://quarto.org/docs/get-started/"
    fi
}

# Main script logic
case "${1:-}" in
    "compile")
        if [[ -z "${2:-}" ]]; then
            echo "❌ Please specify a project/file to compile"
            print_usage
            exit 1
        fi
        compile_latex "$2"
        ;;
    "quarto")
        if [[ -z "${2:-}" ]]; then
            echo "❌ Please specify a project/file to render"
            print_usage
            exit 1
        fi
        render_quarto "$2"
        ;;
    "clean")
        clean_artifacts "${2:-}"
        ;;
    "watch")
        if [[ -z "${2:-}" ]]; then
            echo "❌ Please specify a project/file to watch"
            print_usage
            exit 1
        fi
        watch_latex "$2"
        ;;
    "setup")
        setup_latex
        ;;
    "notebook")
        if [[ -z "${2:-}" ]]; then
            echo "❌ Please specify a notebook to convert"
            print_usage
            exit 1
        fi
        convert_notebook "$2"
        ;;
    *)
        print_usage
        ;;
esac

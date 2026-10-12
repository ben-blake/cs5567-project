# Local build. Needs a TeX distribution with beamer + metropolis.
#   brew install --cask mactex-no-gui    (macOS)
#   sudo apt install texlive-full        (Debian/Ubuntu)
# Or just let CI build it: .github/workflows/slides.yml

midterm.pdf: midterm.tex
	latexmk -xelatex -interaction=nonstopmode midterm.tex

clean:
	latexmk -C
	rm -f *.nav *.snm *.vrb

.PHONY: clean

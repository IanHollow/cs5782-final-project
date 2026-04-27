.PHONY: all pdf check clean distclean

TEX := main.tex
PDF := main.pdf
FINAL := final-poster.pdf
ROOT := ..

all: pdf

pdf:
	nix develop $(ROOT) -c latexmk -g $(TEX)
	cp $(PDF) $(FINAL)

check: pdf
	pdfinfo $(FINAL) | sed -n '1,60p'
	pdffonts $(FINAL) | sed -n '1,40p'

clean:
	nix develop $(ROOT) -c latexmk -c $(TEX)

distclean:
	nix develop $(ROOT) -c latexmk -C $(TEX)
	rm -f main.nav main.snm main.toc main.synctex.gz $(FINAL)

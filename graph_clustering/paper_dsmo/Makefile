.PHONY: clean all exmaples

NAME = 	graph_clustering
REFNAME = references


all: $(NAME).pdf
	- pkill -9 evince
	evince $(NAME).pdf &

$(NAME).pdf : $(NAME).tex $(REFNAME).bib $(REST)
	pdflatex $(NAME).tex
	bibtex $(NAME).aux
	pdflatex $(NAME).tex
	pdflatex $(NAME).tex

clean:
	@rm -rf auto *~ *.aux \
	*.log *.blg *.bbl *.synctex.gz \
	$(NAME).pdf $(NAME).out 

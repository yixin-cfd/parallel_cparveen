TMHOME=/Applications/TeXmacs.app/Contents/MacOS/TeXmacs

TARGET = slides.pdf

all: $(TARGET)

%.pdf: %.tm
	$(TMHOME) -c $*.tm $*.pdf -q

slides.pdf: slides.tm
	$(TMHOME) -c slides.tm slides.pdf -q
	cp slides.pdf ~/Dropbox/Teaching/slides/parallel.pdf

clean:
	rm -rf $(TARGET)

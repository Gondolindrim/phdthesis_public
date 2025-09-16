MAINFILE = main
BUILDDIR = build
OUTPUT_OPTS = max_print_line=400

VIEWER = zathura
VIEWER_OPTS = --fork

TEXENGINE = pdflatex -shell-escape
TEXENGINE_OPTS = --output-directory=${BUILDDIR}


# ANSI terminal colors (see 'man tput') and
# https://linuxtidbits.wordpress.com/2008/08/11/output-color-on-bash-scripts/mppt_pphi/
#
# Don't use color if there isn't a $TERM environment variable:
ifneq ($(strip $(TERM)),)
	BOLD=$(shell tput bold)
	RED=$(shell tput setaf 1)
	GREEN=$(shell tput setaf 2)
	BLUE=$(shell tput setaf 4)
	MAGENTA=$(shell tput setaf 5)
	WHITE=$(shell tput setaf 7)
	UL=$(shell tput sgr 0 1)
	RESET=$(shell tput sgr0 )
endif

# Calling the engine only
pdf:	
	@echo "$(BOLD)Compiling$(RESET) ${MAINFILE}.tex at \"${BUILDDIR}\" folder using ${TEXENGINE} engine ...$(RESET)"
	@$(MAKE) -s check_build
	@$(OUTPUT_OPTS) $(TEXENGINE) $(TEXENGINE_OPTS) $(MAINFILE).tex

view:
	$(VIEWER) $(VIEWER_OPTS) $(BUILDDIR)/$(MAINFILE).pdf

# Compiles compelte document and calls viewer
all: pythonscripts bib view

# Deletes the BUILDDIR
clean: all
	@cp $(BUILDDIR)/${MAINFILE}.pdf ./
	@$(VIEWER) $(VIEWER_OPTS) $(MAINFILE).pdf
	@rm -r $(BUILDDIR)

# Pulls latest from repo
complete:
	@echo "$(BOLD)Pulling latest changes from $(GREEN)master repository$(RESET)"
	git pull origin master
	@$(MAKE) -s bib
	@echo "$(BOLD)Calling viewer..."
	@$(MAKE) -s view

# Checks if BUILDDIR exists. If not, creates it.
check_build:
	@if [ ! -d ${BUILDDIR} ] ; then \
		echo -n "$(BOLD)BUILDDIR check: $(RED)BUILDDIR not found$(WHITE). Creating it...$(RESET)" ; \
		mkdir ${BUILDDIR} ; \
		echo "	$(GREEN)Done.$(RESET)" ; \
	fi

check_results:
	@if [ ! -d ${RESULTDIR} ] ; then \
		echo -n "$(BOLD)RESULTDIR check: $(RED)RESULTDIR not found$(WHITE). Creating it...$(RESET)" ; \
		mkdir ${RESULTDIR} ; \
		echo "	$(GREEN)Done.$(RESET)" ; \
	fi

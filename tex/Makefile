MAINFILE = main
BUILDDIR = build
RESULTDIR = ../scripts/mppt_pphi/result
OUTPUT_OPTS = max_print_line=400

VIEWER = zathura
VIEWER_OPTS = --fork

TEXENGINE = pdflatex -shell-escape
TEXENGINE_OPTS = --output-directory=${BUILDDIR}

IMAGEDIR=../image


TIKZIMAGES := doc-pv_pv doc-pv_iv doc-dvdg doc-didg doc-dvdt doc-didt doc-kt_rsv doc-abs_resistance doc-error_signals doc-mpp_3dcurve doc-p_phi_simplified doc-p_phi_various doc-v_phi_various doc-i_phi_various doc_iv-partial-nodiodes doc_pv-partial-nodiodes doc_iv-partial-diodes doc_pv-partial-diodes doc-vphi_gss_series_array doc-pphi_gss_series_array doc-iphi_gss_series_array doc-rphi_gss_series_array
PYTHONSCRIPTS := '../scripts/mppt_pphi/mpp_tracking/mpp_tracking.py --noplots --savefolder=../scripts/mppt_pphi/results' '../scripts/mppt_pphi/mpp_curve/mpp_curve.py --noplots --savefolder=../scripts/mppt_pphi/results' '../scripts/mppt_pphi/GSS/GSS_static.py --noplots --quiet --savefolder=../scripts/mppt_pphi/results' '../scripts/mppt_pphi/mpp_tracking/mpp_tracking_partial_nodiodes.py --noplots --savefolder=../scripts/mppt_pphi/results' '../scripts/mppt_pphi/mpp_tracking/mpp_tracking_partial_diodes.py --noplots --savefolder=../scripts/mppt_pphi/results' '../scripts/mppt_pphi/GSS/GSS_series_array.py --noplots --savefolder=../scripts/mppt_pphi/results'
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

pythonscripts: check_results
	@printf '$(BOLD)Processing Python scripts...\n$(RESET)' ; \
	for script in  $(PYTHONSCRIPTS) ; do \
		printf '$(GREEN) -->$(RESET) %-100s \n' "$${script}" ; \
		python $${script} ; \
		printf "$(BOLD)$(GREEN)Done.$(RESET)\n" ; \
	done

# Generates a particular TikZ job. Usage: make tikzjob JOB=<jobname>
tikzjob: check_build
	@echo "Processing TikZ job ${JOB}..." ; \
	printf '$(GREEN) -->$(RESET) %-100.100s' $${JOB} ; \
	$(OUTPUT_OPTS) $(TEXENGINE) --jobname=$${JOB} $(TEXENGINE_OPTS) '\let\GenerateTikzJobs=Y\input{$(MAINFILE).tex}' ; \
	printf "$(GREEN)Done.$(RESET)\n" ; \


# Generates TikZ images. The exact instructions are in the TeX file.
tikzimages: check_build
	@echo "Processing tikzimage jobs..." ; \
	for job in $(TIKZIMAGES); do \
		printf '$(GREEN) -->$(RESET) %-100.100s' $${job} ; \
		$(OUTPUT_OPTS) $(TEXENGINE) --jobname=$${job} $(TEXENGINE_OPTS) '\let\GenerateTikzJobs=Y\input{$(MAINFILE).tex}' ; \
		printf "$(GREEN)Done.$(RESET)\n" ; \
	done

# Generates a single TikZ image; useful when editing images themselves. How to use: make tikzimage images='<name>'
tikzlist: check_build
	@echo "Processing tikzimage jobs..." ; \
	for job in $(IMAGES); do \
		printf '$(GREEN) -->$(RESET) %-100.100s' $${job} ; \
		$(OUTPUT_OPTS) $(TEXENGINE) --jobname=$${job} $(TEXENGINE_OPTS) '\let\GenerateTikzJobs=Y\input{$(MAINFILE).tex}' ; \ # > /dev/null ; \
		printf "$(GREEN)Done.$(RESET)\n" ; \
	done


# Calling the engine only
pdf:	
	@echo "$(BOLD)Compiling$(RESET) ${MAINFILE}.tex at \"${BUILDDIR}\" folder using ${TEXENGINE} engine ...$(RESET)"
	@$(MAKE) -s check_build
	@$(OUTPUT_OPTS) $(TEXENGINE) $(TEXENGINE_OPTS) $(MAINFILE).tex

2pdf: pdf
	$(MAKE) -s pdf

# Calls the engine once, compiles bibliography and runs engine twice
bib: 	
	@echo "$(BOLD)Compiling $(GREEN)${MAINFILE}.tex$(WHITE) at $(MAGENTA)$(BOLD)\"${BUILDDIR}\"$(WHITE) folder using $(BLUE)${TEXENGINE} $(WHITE) engine ...$(RESET)"
	@echo '$(GREEN) -->$(RESET) Compiling first run...'
	@$(MAKE) -s pdf > /dev/null
	@echo '$(GREEN) -->$(RESET) Compiling bibliography...'
	@bibtex $(BUILDDIR)/$(MAINFILE) > /dev/null ||:
	@echo '$(GREEN) -->$(RESET) Compiling second run...'
	@$(MAKE) -s pdf > /dev/null
	@echo '$(GREEN) -->$(RESET) Compiling third run...'
	@$(MAKE) -s pdf > /dev/null

# Calls PDF viewer
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

% -------------------------------------------------------------------
% APA-Style thesis LaTeX Template
% AUTHOR: Álvaro "Gondolindrim" Volpato (alvaro.volpato@usp.br)
% VERSION: 1.1
% PAGE: http://github.com/Gondolindrim/apaThesis
% LICENSE: Creative-Commons Non-commercial Share-Alike
%--------------------------------------------------------------------

% This is a class file for a thesis that fits the American Phychology
% Association standards, described in 

% American Psychology Association (2009). Publication Manual of the American Psychological Association, sixth edition.

% It is supposed to be used as a template for a masters or PhD thesis. It was written in VIM and it contains
% folding data (the three "{") in the text. To enable this folding, type in vim:

% :setfoldmethod=marker

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{apaThesis}[2019/04/07 APA Thesis Class by Alvaro Augusto Volpato]

\LoadClass[
	11pt,				% Font size
	openright,			% Chapters start in odd page
	twoside,			% Printing setting as double-sided document
	a4paper,			% Paper size
	english,			% Main language
	brazil				% Additional languages
	]{memoir}

% -------------------------------------------------
% TABLE OF CONTENTS
% -------------------------------------------------
% (0) Page geometry: paper size, margins
% (1) Document coding: inputenc and fontenc
% (2) Font configuration: math font, sans, serif and body fonts
% (3) Mathematical packages: formulas, custom operators
% (4) Hyperlinks
% (5) Graphics packages: grpahics, colors, svg and mathcolor command
% (6) Headers and footers package
% (7) Customizing headers
% (8) Customizing lists: LoC, LoT, LoF, LoA, LoS
% (9) Customizing tables and figures captions
% (11) Configuring APA-Style citations
% (12) Configuring listings (programs and codes)
% (13) Newabstract environment for multiple abstracts
% (14) Title page configuration
% (15) Front matter and multiple front matters
% (16) Document data - author, title, place, date, advisor
% (17) Epigraph - the nice sounding statement you put there to sound smart
% (18) pcite command
% -------------------------------------------------

% -----------------------------------------------
% (0) PAGE GEOMETRY {{{1
% -----------------------------------------------
\usepackage[a4paper, top=1in,bottom=1in,left=1in,right=1in]{geometry}

% -----------------------------------------------
% (1) INPUTENC AND FONTENC CODING {{{1
% -----------------------------------------------
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}

% -----------------------------------------------
% (2) MAIN FONTS {{{1
% -----------------------------------------------

\usepackage{times}		% Loads the Times-Roman Fonts
\usepackage{inconsolata}

%\usepackage{PTSans}
\usepackage{Rosario}
%\usepackage{sourcesanspro}

\usepackage{lettrine}	% First capitalized letter at every chapter

\usepackage{ragged2e}

% -----------------------------------------------
% (3) MATHEMATICAL PACKAGES {{{1
% -----------------------------------------------

\usepackage[framemethod=tikz]{mdframed}

\newcommand{\xfrac}[4]{\genfrac{}{}{}{}{\raisebox{#1}{$#3$}}{\raisebox{-#2}{$#4$}} }
\usepackage{csvsimple}

\usepackage{amsthm}
\usepackage[gelasio]{newtxmath}		% Loads the Times-Roman Math Fonts
\usepackage{mathtools}
\usepackage{thmtools}
\usepackage[svgnames,table]{xcolor}
\usepackage{mdframed}
\usepackage{esint}

\newtheoremstyle{highlighted}% 
{}{}% 
{}{}% 
{}{}% 
{5mm}% 
{\colorbox{black!15}{\textbf{\thmname{#1} \thmnumber{#2}} \thmnote{(#3)}. \hspace{\textwidth - 2mm - \widthof{\thmname{#1}\thmnumber{#2}\thmnote{(#3)} }} }\vspace{3mm}}

\declaretheoremstyle[headpunct={}, bodyfont=\itshape]{italicbody}
\declaretheorem[style=italicbody]{definition}

\declaretheoremstyle[headpunct={} ]{defaulttheorem}
\declaretheorem[style=defaulttheorem]{theorem,lemma}
\declaretheorem[name = Corollary, numberwithin=theorem, style=shadedtheorem]{corollary}
\declaretheorem[name = Example, style=highlighted]{example}
\newcommand{\examplebar}{\vspace{3mm} \noindent \colorbox{black!15}{\raisebox{3mm}{}  \hspace{\textwidth}}}
\declaretheorem[name = Note, numberwithin=theorem]{note}
\declaretheorem[name = Notation, numbered=no]{notation}

\declaretheorem[name = Remark T\hspace{-0.75mm}, numberwithin=theorem]{remark}
\declaretheorem[name = Remark D\hspace{-0.75mm}, numberwithin=definition]{definitionremark}

\allowdisplaybreaks
\usepackage{cancel}
\renewcommand{\Re}{\operatorname{Re}}
\renewcommand{\Im}{\operatorname{Im}}
\newcommand{\Res}{\operatorname{Res}}

\usepackage{xfrac}
\usepackage{algorithm}
\usepackage{algpseudocode}

% For evenly spaced columns in matrices
\newcolumntype{C}[1]{>{\centering\arraybackslash$}m{#1}<{$}}
\newlength{\mycolwd}                                         % array column width
\setlength{\mycolwd}{5cm}% "width" of $e^{-\frac{i}{\hbar}|A|t$; largest element in array

\DeclareMathOperator{\sat}{sat}
\DeclareMathOperator{\tr}{tr}
\DeclareMathOperator{\sign}{sign}
\DeclareMathOperator{\supp}{supp}
\DeclareMathOperator{\mathnull}{null}
\DeclareMathOperator{\mathspan}{span}
\newcommand*{\hermitian}{{\mathsf{H}}}

\newenvironment{Array}[2]{
	\begin{array}{*{#1}{@{} C {#2}@{} }}
	}
	{
	\end{array}
}

% To make subequations
\usepackage{empheq}

% To make big sum symbols
\newcommand{\dsum}{\displaystyle\sum}

% To show cancelling terms in equations
\usepackage{cancel}

% To show rotated equations
\usepackage{lscape}

% Defining \dashint, \lowdashint, \highdashint for Cauchy Principal Value
% Source: https://tex.stackexchange.com/questions/204547/integral-symbol-with-high-and-low-dashes
\def\Xint#1{\mathchoice
   {\XXint\displaystyle\textstyle{#1}}%
   {\XXint\textstyle\scriptstyle{#1}}%
   {\XXint\scriptstyle\scriptscriptstyle{#1}}%
   {\XXint\scriptscriptstyle\scriptscriptstyle{#1}}%
   \!\int}
\def\XXint#1#2#3{{\setbox0=\hbox{$#1{#2#3}{\int}$}
     \vcenter{\hbox{$#2#3$}}\kern-.5\wd0}}
\def\ddashint{\Xint=}
\def\dashint{\Xint-}

% For circled numbers
\usepackage{enumitem}

% -----------------------------------------------
% (4) SETTING HYPERLINKS {{{1
% -----------------------------------------------
\usepackage{hyperref}
\makeatletter
\hypersetup{
	colorlinks=false,       		% false: boxed links; true: colored links
	pdfborder = {0 0 0},
	linkbordercolor = {1 1 1}
}
\makeatother

% Declaring custom operators
\DeclareMathOperator{\im}{Im}
\DeclareMathOperator{\re}{Re}

% -----------------------------------------------
% (5) GRAPHICS PACKAGES {{{1
% -----------------------------------------------

\usepackage{xcolor}
\usepackage{graphicx}
%\graphicspath{{./images/}}
\definecolor{bluekeywords}{rgb}{0.13,0.13,1}
\definecolor{greencomments}{rgb}{0,0.5,0}
\definecolor{redstrings}{rgb}{0.9,0,0}
\definecolor{DarkGreen}{rgb}{0.0,0.4,0.0}	% Comment color
\definecolor{highlight}{RGB}{250,250,240}	% Code highlight color
\definecolor{Gray}{RGB}{120,120,120}		% Code highlight color
\definecolor{Blue}{RGB}{0,0,255}
\definecolor{Purple}{RGB}{255,0,255}
\definecolorset{RGB}{a}{}{Pink,200,0,180;Blue,0,180,250;Green,50,180,50;Yellow,250,150,0;Red,250,0,100}
\definecolorset{RGB}{stewart}{}{blue,0,180,250;pink,250,0,180;yellow,250,180,0;green,0,180,0;purple,120,0,250}
\definecolorset{RGB}{python}{}{blue,21,119,180;yellow,255,127,14;green,44,160,44;red,214,39,40}

% Mathcolor command used to color parts of formulas. Code snippet obrained at
% https://tex.stackexchange.com/questions/21598/how-to-color-math-symbolsa
% Usage: \mathcolor{<color>}{<formula>}
\makeatletter
\def\mathcolor#1#{\@mathcolor{#1}}
\def\@mathcolor#1#2#3{%
  \protect\leavevmode
  \begingroup
    \color#1{#2}#3%
  \endgroup
}
\makeatother

\usepackage{wrapfig}

% To be able to load SVGs. Needs inkscape installed and --shell-escape for pdflatex
\usepackage[inkscapearea=page]{svg}

% -----------------------------------------------
% (6) FANCYHDR PACKAGE {{{1
% -----------------------------------------------

\usepackage{fancyhdr}
\fancyhf{}
%\pagestyle{fancy}
%\fancyhead[LE]{\small\sffamily \nouppercase{\rightmark}}
%\fancyhead[RE]{\sffamily \thepage}
%\fancyhead[LO]{\sffamily \thepage}
%\fancyhead[RO]{\small\sffamily \nouppercase{\leftmark}}

%\renewcommand{\chaptermark}[1]{\markboth{\thechapter.~#1}{}}

% Page styles
% Plain used in the abstracts
\fancypagestyle{abstractplain}{%
	\fancyhf{}
	\renewcommand{\headrulewidth}{0cm}
	\renewcommand{\headsep}{3cm}
	\fancyfoot[C]{\sffamily \thepage}
}


\fancypagestyle{plain}{%
	\fancyhf{}
	\renewcommand{\headrulewidth}{0cm}
	\fancyfoot[C]{\sffamily \thepage}
}

% body style used in the text body
\fancypagestyle{body}{%
	\fancyhf{}
	\fancyhead[RE]{\small\sffamily \nouppercase{\rightmark}}
	\fancyhead[LE]{\sffamily \thepage}
	\fancyhead[RO]{\sffamily \thepage}
	\fancyhead[LO]{\small\sffamily \nouppercase{\leftmark}}
}

% -----------------------------------------------
% (7) CUSTOMIZING HEADERS {{{1
% -----------------------------------------------
\usepackage[sc]{titlesec}

% (7.1) PART
\renewcommand{\thepart}{\Roman{part}}
\titleformat{\part}[display]
{\sffamily\bfseries\Large}
{\filleft \MakeUppercase{\normalfont\small\sffamily\partname} \Huge\thepart}
{2mm}
{\titlerule
\vspace{5mm}%
\filright}
[\vspace{5mm}%
\titlerule]

% (7.2) PART

\renewcommand{\thepart}{\arabic{part}}
\titleformat{\part}[display]
{\pagestyle{empty}\bfseries\sffamily\Large}
{\filleft \MakeUppercase{\normalfont\small\sffamily\partname} \Huge\thepart}
{2mm}
{\titlerule
\vspace{5mm}%
\filright}
[\vspace{5mm}%
\titlerule]

%\titleclass{\part}{page}
%\assignpagestyle{\part}{empty}
%\titleformat{\part}
%    [display]
%    {\centering\sffamily\Huge}
%    {\partname\ \thepart}
%    {0pt}
%    {\huge}
%    [\clearpage]

% (7.2) CHAPTER
\renewcommand{\thechapter}{\arabic{chapter}}
\titleformat{\chapter}[display]
{\bfseries\sffamily\Large}
{\filleft \MakeUppercase{\normalfont\small\sffamily\chaptertitlename} \Huge\thechapter}
{2mm}
{\titlerule
\vspace{5mm}%
\filright}
[\vspace{5mm}%
\titlerule]

% (7.3) SECTION
\titleformat{\section}[block]
{\normalfont\bfseries\sffamily}
{\hspace{5mm}\thesection}{.5em}{\bfseries}

% (7.4) SUBSECTION
\titleformat{\subsection}[block]
{\itshape\bfseries\sffamily}
{\hspace{1cm}\thesubsection}{.5em}{}

% (7.5) SUBSUBSECTION
\titleformat{\subsubsection}[block]
{\itshape\sffamily}
{}{.5em}{}

% -----------------------------------------------
% (8) CUSTOMIZING ToC, LoF, LoT, LoS, LoA {{{1
% -----------------------------------------------

% The pre-textual elements have roman numbering
\AtBeginDocument{
	\pagestyle{body}
	\pagenumbering{arabic}
}

% (8.1) ToC {{{2 --------------------------------

% Setting depth
\setcounter{tocdepth}{4}
\setcounter{secnumdepth}{2}

% Defining new color
\definecolor{gray}{RGB}{150,150,150}

% Defining indents
\renewcommand*{\cftappendixname}{APPENDIX\space}
\setlength{\cftchapterindent}{5mm}
\setlength{\cftsectionindent}{10mm}
\setlength{\cftsubsectionindent}{15mm}
\setlength{\cftsubsubsectionindent}{20mm}
\setlength{\cftsectionnumwidth}{2em}
\setlength{\cftsubsectionnumwidth}{2.5em}
\setlength{\cftsubsubsectionnumwidth}{2.5em}
\renewcommand{\contentsname}{\sffamily Contents}
\renewcommand{\cftsectionpresnum}{\bfseries} % Bold face for section numbers

% Elimitating dots
\renewcommand{\cftdot}{.}
\renewcommand{\cftchapterdotsep}{\cftnodots}
\renewcommand{\cftsectiondotsep}{\cftdotsep}
\renewcommand{\cftsubsectiondotsep}{\cftdotsep}
\renewcommand{\cftsubsubsectiondotsep}{\cftnodots}

\makeatletter
% Editing part line
\renewcommand\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{-\@highpenalty}%
    \addvspace{2.25em \@plus\p@}%
    \setlength\@tempdima{3em}%
    \begingroup
      \parfillskip -\@pnumwidth
      \centering{\leavevmode
       \large\bfseries\sffamily #1 }\par
       \nobreak
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
    \endgroup
  \fi}

% Editing chapter line
\let\stdl@chapter\l@chapter
\renewcommand*{\l@chapter}[2]{%
%	\stdl@chapter{\small\sffamily\textcolor{gray}{#1} }{\sffamily\textcolor{gray}{#2}}	% Gray color
	\stdl@chapter{\small\sffamily #1 }{\small\sffamily #2}	% Black color
}
% Making chapter titles in full uppercase 
\let\oldcontentsline\contentsline
\def\contentsline#1#2{%
  \expandafter\ifx\csname l@#1\endcsname\l@chapter
    \expandafter\@firstoftwo
  \else
    \expandafter\@secondoftwo
  \fi
  {%
    \oldcontentsline{#1}{\MakeTextUppercase{#2}}%
  }{%
    \oldcontentsline{#1}{#2}%
  }%
}

% Editing appendix line
\let\stdl@appendix\l@appendix
\renewcommand*{\l@appendix}[2]{%
%	\stdl@appendix{\small\sffamily\textcolor{gray}{#1}}{\sffamily\textcolor{gray}{#2}}	% Gray color
	\stdl@appendix{\small\sffamily #1}{\small\sffamily #2}
}

% Editing section line
\let\stdl@section\l@section
\renewcommand*{\l@section}[2]{%
  \stdl@section{\small\bfseries\sffamily\textcolor{gray}{#1}}{\small\sffamily\bfseries\textcolor{gray}{#2} }
}

% Editing subsection line
\let\stdl@subsection\l@subsection
\renewcommand*{\l@subsection}[2]{%
  \stdl@subsection{\footnotesize\sffamily\bfseries\itshape{#1}}{\small\sffamily{#2}}
}

% Editing subsubsection line
\let\stdl@subsubsection\l@subsubsection
\renewcommand*{\l@subsubsection}[2]{%
 \stdl@subsubsection{\footnotesize\sffamily\itshape{#1}}{}
}
\makeatother


% (8.2) LoT, LoF {{{2 ---------------------------
% Customizing LoT and LoF headers

\makeatletter
% Editing figure line
\let\stdl@figure\l@figure
\renewcommand*{\l@figure}[2]{%
 \stdl@figure{\footnotesize\sffamily\itshape{#1}}{\footnotesize\sffamily\itshape{#2}}
}

% Editing table line
\let\stdl@table\l@table
\renewcommand*{\l@table}[2]{%
 \stdl@table{\footnotesize\sffamily\itshape{#1}}{\footnotesize\sffamily\itshape{#2}}
}
\makeatother

\renewcommand{\printloftitle}{\centering\HUGE\sffamily\bfseries}
\renewcommand{\printlottitle}{\centering\HUGE\sffamily\bfseries}

% (8.3) LoA {{{2 --------------------------------
% Defining new \newitem command to customize appearance of List of Acronyms

	\newcommand{\acronym}[2]{\item[{\sffamily #1}] \textit{#2}}

% Defining acronyms environment

\newenvironment{acronyms}{
	\thispagestyle{plain}
	\addcontentsline{toc}{chapter}{Abbreviations and Acronyms}	% Add to ToC
	\chapter*{\vspace*{0.6in} Abbreviations and Acronyms}		% Add unnumbered chapter
	\begin{symbols}							% Begins symbol listing
}{
	\end{symbols}
	\cleardoublepage
}

% Defining acronyms environment
\newenvironment{listofsymbols}{
	\thispagestyle{plain}
	\addcontentsline{toc}{chapter}{List of Symbols}			% Add to ToC
	\chapter*{\vspace*{0.6in}  List of Symbols}			% Add unnumbered chapter
	\begin{symbols}							% Begins symbol listing
}{
	\end{symbols}
	\cleardoublepage
}

% Defining acronyms environment
\newenvironment{judgepage}{
	\cleardoublepage
	\thispagestyle{empty}
{
}
	\begin{figure}[h]
	\centering
	\vspace*{-2cm}
	\makebox[\linewidth]{
	\includegraphics[height = 1\paperheight]{../images/folha_julgamento.pdf}
}
	\end{figure}
	\cleardoublepage
}

% Defining acronyms environment
\newenvironment{customfrontmatter}{
	\thispagestyle{empty}
{
}
	\begin{figure}[h]
	\centering
	\vspace*{-2.6cm}
	\makebox[\linewidth]{
	\includegraphics[height = 1.01\paperheight]{../images/customfrontmatter.pdf}
}
	\end{figure}
	\cleardoublepage
}

\newenvironment{customendmatter}{
	\thispagestyle{empty}
{
}
	\begin{figure}[h]
	\centering
	\vspace*{-2.6cm}
	\makebox[\linewidth]{
	\includegraphics[height = 1.01\paperheight]{../images/customendmatter.pdf}
}
	\end{figure}
	\cleardoublepage
}

% -----------------------------------------------
% (9) CUSTOMIZING TABLE AND FIGURE CAPTIONS AND NUMBERING {{{1
% -----------------------------------------------
\usepackage{caption}
\usepackage{tabularx}	% For tables that wpan the whole textwidth

\usepackage{chngcntr}	% Making numbering continuous
\counterwithout{figure}{chapter}
\counterwithout{table}{chapter}

% Setting up table captions
\captionsetup[table]{
  labelfont={sf,sc},
  textfont={small},
  labelsep=newline,singlelinecheck=false,format=plain,
  parindent=1em}

% Setting up table captions
\captionsetup[figure]{
  labelfont={sf,it},
  textfont={small},
  labelsep=period,singlelinecheck=false,format=plain,
  parindent=1em}

% -----------------------------------------------
% (11) APA STYLE CITATION {{{1
% -----------------------------------------------

\usepackage{natbib}
\bibliographystyle{apalike}

% Removing obsolete estra spacing between lines and periods
\frenchspacing 

% -----------------------------------------------
% (12) LISTINGS CONFIGURATION {{{1
% -----------------------------------------------

\usepackage{listings}
\lstdefinestyle{Style1}{ % Define a style for your code snippet, multiple definitions can be made if, for example, you wish to insert multiple code snippets using different programming languages into one document
language=Python, % Detects keywords, comments, strings, functions, etc for the language specified
backgroundcolor=\color{highlight}, % Set the background color for the snippet - useful for highlighting
basicstyle=\tiny\ttfamily, % The default font size and style of the code
breakatwhitespace=false, % If true, only allows line breaks at white space
breaklines=true, % Automatic line breaking (prevents code from protruding outside the box)
captionpos=t, % Sets the caption position: b for bottom; t for top
commentstyle=\color{DarkGreen}\ttfamily, % Style of comments within the code - dark green courier font
deletekeywords={}, % If you want to delete any keywords from the current language separate them by commas
%escapeinside={\%}, % This allows you to escape to LaTeX using the character in the bracket
firstnumber=1, % Line numbers begin at line 1
frame=single, % Frame around the code box, value can be: none, leftline, topline, bottomline, lines, single, shadowbox
frameround=tttt, % Rounds the corners of the frame for the top left, top right, bottom left and bottom right positions
keywordstyle=\color{Blue}\bfseries\ttfamily, % Functions are bold and blue
morekeywords={}, % Add any functions no included by default here separated by commas
numbers=left, % Location of line numbers, can take the values of: none, left, right
numbersep=10pt, % Distance of line numbers from the code box
numberstyle=\tiny\color{Gray}, % Style used for line numbers
rulecolor=\color{black}, % Frame border color
showstringspaces=false, % Don't put marks in string spaces
showtabs=false, % Display tabs in the code as lines
stepnumber=5, % The step distance between line numbers, i.e. how often will lines be numbered
stringstyle=\color{Purple}, % Strings are purple
tabsize=2, % Number of spaces per tab in the code
}

% -----------------------------------------------
% (13) NEWABSTRACT ENVIRONMENT {{{1
% -----------------------------------------------

% Building the newabstract environment, which enables usage of various abstracts in more than one language
%\renewcommand{\abstractnamefont}{\chaptitlefont}
\newenvironment{newabstract}[1]{
	\thispagestyle{abstractplain}
	\renewcommand{\abstractname}{#1}	% Rename abstract
	\addcontentsline{toc}{chapter}{#1}	% Add to ToC with new name
	\begin{abstract}}{
	\end{abstract}\cleardoublepage\newpage}

% -----------------------------------------------
% (14) TITLE PAGE {{{1
% -----------------------------------------------

\newcommand{\printtitlepage}{
	\begin{titlingpage}

	%	\vspace*{1cm}

		\center\includegraphics[width=0.2\textwidth]{../images/uniLogo.pdf}
		\vfill
		
		\begin{minipage}{0.75\textwidth}
			\centering
			\bfseries\large\sffamily\thetitle
		\end{minipage}
				
		\vfill

		\large\sffamily\theauthor

		\vfill

		\theplace

		\thewritingdate

		\vspace*{1cm}
	\end{titlingpage}\cleardoublepage
}

% -----------------------------------------------
% (15) FRONT MATTER {{{1
% -----------------------------------------------
\newcommand{\printfrontmatter}{
	\begin{center}
	\thispagestyle{empty}

		\vspace*{5cm}

		\sffamily\bfseries\large\thetitle
		\sffamily\normalfont\normalsize
		\vspace*{\fill}
		
		% Printing preamble
		\begin{minipage}[t]{0.7\textwidth}
			\sffamily\center\normalsize\thepreamble

			\vspace{1cm} by\par \sffamily\theauthor

			\vspace*{1cm} Advisor: \sffamily\theadvisor
		\end{minipage}

		\vspace*{\fill}

		\sffamily\theaffiliation

		\vspace*{\fill}

		\sffamily\theplace
		\par
		\sffamily\thewritingdate
		\vspace*{1cm}
	\end{center}
%	\cleardoublepage
}

\newcommand{\anotherfrontmatter}[7]{
	\begin{center}
	\thispagestyle{empty}

		\vspace*{5cm}

		{\sffamily\bfseries\large #1}
		\sffamily\normalfont\normalsize
		\vspace*{\fill}
		
		% Printing preamble
		\begin{minipage}[t]{0.7\textwidth}
			\sffamily\center\normalsize #7

			\vspace{1cm} \par \sffamily #2

			\vspace*{1cm} Orientador: \sffamily #6
		\end{minipage}

		\vspace*{\fill}

		{\sffamily #5}

		\vspace*{\fill}

		{\sffamily #3}
		\par
		{\sffamily #4}
		\vspace*{1cm}
	\end{center}
	\cleardoublepage
}


% -----------------------------------------------
% (16) DOCUMENT DATA {{{1
% -----------------------------------------------

% Title and author
\title{The APA thesis class file}
\author{Álvaro Augusto Volpato}

% Place
\newcommand{\theplace}{}
\newcommand{\place}[1]{\renewcommand{\theplace}{#1}}
\place{São Carlos, Brazil}

% Writingdate
\newcommand{\thewritingdate}{}
\newcommand{\writingdate}[1]{\renewcommand{\thewritingdate}{#1}}
\writingdate{May 2019}

% Affiliation
\newcommand{\theaffiliation}{}
\newcommand{\affiliation}[1]{\renewcommand{\theaffiliation}{#1}}
\affiliation{%
  Universidade de São Paulo
  \par
  São Carlos School of Engineering
  \par
  Department of Electrical and Computer Engineering}

% Advisor
\newcommand{\theadvisor}{}
\newcommand{\advisor}[1]{\renewcommand{\theadvisor}{#1}}
\advisor{Associate Professor Dr. Luís Fernando Costa Alberto}

% Preamble
\newcommand{\thepreamble}{}
\newcommand{\preamble}[1]{\renewcommand{\thepreamble}{#1}}
\preamble{ Apa Thesis Class file example and documentation}

\newcommand{\begintextbody}{
	\pagestyle{body}
}

% -----------------------------------------------
% (17) EPIGRAPH {{{1
% -----------------------------------------------

\newenvironment{newepigraph}{
	\par\vspace*{\fill}
	\thispagestyle{empty}
	\hfill
	\begin{minipage}[t]{0.5\textwidth}\itshape
}{ 
	\end{minipage}
	\clearpage
}

% -----------------------------------------------
% (18) pcite COMMAND {{{1
% -----------------------------------------------

% This command is used to cite sources with two parantheses (<citation>)
% specially when author-date system is used

\newcommand{\pcite}[1]{ (\cite{#1})~\hspace{-0.8mm}}

\newcommand\scalemath[2]{\scalebox{#1}{\mbox{\ensuremath{\displaystyle #2}}}}
% -----------------------------------------------
% (13) ACKNOWLEDGEMENTS ENVIRONMENT {{{1
% -----------------------------------------------

% Building the newabstract environment, which enables usage of various abstracts in more than one language
\newenvironment{acknowledgements}{
	\thispagestyle{abstractplain}
	\renewcommand{\abstractname}{Acknowledgements}	% Rename abstract
	\addcontentsline{toc}{chapter}{\abstractname}	% Add to ToC with new name
	\begin{abstract}}{
	\end{abstract}\cleardoublepage\newpage}

%For rotated tables
\usepackage{rotating}

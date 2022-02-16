C=======================================================================
C
C  DESCRIPTION OF THE VARIABLES CONTAINED IN THE COMMONS
C  =====================================================================
C  (IN ALPHABETIC ORDER OF THE COMMON)
C
C
C -----------------------------/CHRATE/---------------------------------
C  FACTOR(NREC)     = Multiplicative factor for the rate of reaction i-th
C  HCHRAT(NREC)     = Type of changes adopted for reaction i-th
C  NCHRAT           = Number of reactions to be changed
C  WCHRAT(NREC)     = Reactions to be changed
C
C -----------------------------/ERRHANDLE/------------------------------
C  EFLAG            = Flag for the completion of the run
C                     EFLAG=0 successfull run
C                     EFLAG=1 repeat run with a small perturbation
C                     EFLAG=2 interrupt running
C  EMESS            = Error message
C
C -----------------------------/INPCARD/--------------------------------
C  CMODE            = Flag for the choice of the running mode
C  FOLLOW           = Option for following the evolution on the screen
C                     (card mode)
C  OVERW            = Option for overwriting the output files (card mode)
C
C -----------------------------/NETWRK/---------------------------------
C  INUC             = Number of nuclides in the selected network
C  IREC             = Number of reactions among nuclides in the selected
C                     network
C  IXT(30)          = Code of the nuclides whose evolution has to be
C                     followed (ixt(30)=control integer)
C  NVXT             = Number of nuclides whose evolution has to be
C                     followed
C
C -----------------------------/NSYMB/----------------------------------
C  BYY(NNUC+1)      = Text strings for the output
C
C -----------------------------/OUTFILES/-------------------------------
C  NAMEFILE1        = Name of the output file with the final values of
C                     the nuclide abundances
C  NAMEFILE2        = Name of the output file with the evolution of the
C                     nuclides whose evolution has to be followed
C  NAMEFILE3        = Name of the output file with the information on the
C                     numerical integration
C
C -----------------------------/READINP/--------------------------------
C  CFLAG            = Flag for the input variable type in the card
C                     reading (card mode)
C  IKEY             = Progressive argument key number in the card reading
C                     (card mode)
C  ISTART           = Starting point of the line in the card reading
C                     (card mode)
C  DNCHRAT          = Number of reactions to be added to the changed ones
C  LINE             = Line input from the card file (card mode)
C
C -----------------------------/RSTRINGS/-------------------------------
C  RSTRING(NREC)    = Reaction text strings
C
C=======================================================================
C
C  DESCRIPTION OF THE VARIABLES CONTAINED IN THE COMMONS
C  =====================================================================
C  (IN ALPHABETIC ORDER OF THE VARIABLE NAMES)
C
C
C  BYY(NNUC+1)      = Text strings for the output
C  CFLAG            = Flag for the input variable type in the card
C                     reading (card mode)
C  CMODE            = Flag for the choice of the running mode
C  DNCHRAT          = Number of reactions to be added to the changed ones
C  EFLAG            = Flag for the completion of the run
C                     EFLAG=0 successfull run
C                     EFLAG=1 repeat run with a small perturbation
C                     EFLAG=2 interrupt running
C  EMESS            = Error message
C  FACTOR(NREC)     = Multiplicative factor for the rate of reaction i-th
C  FOLLOW           = Option for following the evolution on the screen
C                     (card mode)
C  HCHRAT(NREC)     = Type of changes adopted for reaction i-th
C  IKEY             = Progressive argument key number in the card reading
C                     (card mode)
C  INUC             = Number of nuclides in the selected network
C  IREC             = Number of reactions among nuclides in the selected
C                     network
C  ISTART           = Starting point of the line in the card reading
C                     (card mode)
C  IXT(30)          = Code of the nuclides whose evolution has to be
C                     followed (ixt(30)=control integer)
C  LINE             = Line input from the card file (card mode)
C  NAMEFILE1        = Name of the output file with the final values of
C                     the nuclide abundances
C  NAMEFILE2        = Name of the output file with the evolution of the
C                     nuclides whose evolution has to be followed
C  NAMEFILE3        = Name of the output file with the information on the
C                     numerical integration
C  NCHRAT           = Number of reactions to be changed
C  NVXT             = Number of nuclides whose evolution has to be
C                     followed
C  OVERW            = Option for overwriting the output files (card mode)
C  RSTRING(NREC)    = Reaction text strings
C  WCHRAT(NREC)     = Reactions to be changed
C
C=======================================================================
	PROGRAM MAIN
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	Main program used to choose the input parameters and customize the
C	  output
C
C	Calls to READLINE, READRATES, and PARTHENOPE
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	IMPLICIT DOUBLE PRECISION (A-Z)
C--------------------------Local variables------------------------------
	INTEGER          I,J,ISEL,ICHRAT
	CHARACTER        CSEL*1
	LOGICAL          FSTIME
C-----Network parameters
	INTEGER          NNUC,NREC
	PARAMETER        (NNUC=26,NREC=100)
C-----Changing rate parameters
	CHARACTER*7      CHOPT(4)
	DATA             CHOPT/'ADOPTED','LOW','HIGH','FACTOR'/
C-----Reaction excluded by default
	INTEGER          REXCLUDED(NREC)
	DATA             FSTIME/.TRUE./
C-----Parameters for the input card mode
	CHARACTER        KEYWORD*9,VALC*500
	LOGICAL          VALL,IERROR
	CHARACTER*500    INPUTF
C-----BBN evolution
	EXTERNAL         PARTHENOPE
C--------------------------Common variables-----------------------------
	INTEGER          WCHRAT(NREC),HCHRAT(NREC),NCHRAT
	DIMENSION        FACTOR(NREC)
	COMMON/CHRATE/   WCHRAT,HCHRAT,FACTOR,NCHRAT

	INTEGER          EFLAG
	CHARACTER*50     EMESS
	COMMON/ERRHANDLE/EFLAG,EMESS

	CHARACTER        CMODE*1
	LOGICAL          FOLLOW,OVERW
	COMMON/INPCARD/  FOLLOW,OVERW,CMODE

	INTEGER          INUC,IREC,NVXT,IXT(30)
	COMMON/NETWRK/   INUC,IREC,NVXT,IXT

	CHARACTER*5      BYY(NNUC+1)
	COMMON/NSYMB/    BYY

	CHARACTER*500    NAMEFILE1,NAMEFILE2,NAMEFILE3
	COMMON/OUTFILES/ NAMEFILE1,NAMEFILE2,NAMEFILE3

	INTEGER          ISTART,IKEY,DNCHRAT
	CHARACTER        CFLAG*1,LINE*500
	COMMON/READINP/  ISTART,IKEY,DNCHRAT,CFLAG,LINE

	CHARACTER*14     RSTRING(NREC)
	COMMON/RSTRINGS/ RSTRING
C-----------------------------------------------------------------------

C-----Choose between interactive or input card mode
	write(*,*) 'Please, choose between interactive or input card ',
     .    'mode'
	write(*,*) '(i or c?)'
1	read(*,*) cmode
	write(*,*) cmode
	if (cmode.eq.'c') then
	  goto 2
	elseif (cmode.eq.'i') then
	  goto 3
	else
	  write(*,999)
	  goto 1
	endif

C----------------------------Input card mode---------------------------
C-----Default values for all run parameters
2	goto 800
C-----Open input card
21	write(*,*) 'Please, enter the name of the input card file'
	read(*,'(a)') inputf
	write(*,*) inputf
	open(unit=1,file=inputf,status='old')
C-----Get an input line
22	read(1,'(a)') line
	istart=0
	if (line(1:5).eq.'ETA10') then
	  keyword='ETA10'
	  ikey=1
	  ierror=.false.
	  cflag='d'
	  call readline('ETA10',valr,vall,valc,ierror)
	  if (ierror) then
	    write(*,99) keyword
	    stop
	  endif
	  eta10=valr
	  if (eta10.lt.2.d0 .or. eta10.gt.9.d0) then
	    write(*,99) keyword
	    stop
	  endif
	elseif (line(1:4).eq.'DNNU') then
	  keyword='DNNU'
	  ikey=1
	  ierror=.false.
	  cflag='d'
	  call readline('DNNU',valr,vall,valc,ierror)
	  if (ierror) then
	    write(*,99) keyword
	    stop
	  endif
	  dnnu0=valr
	  if (dnnu0.lt.-3.d0 .or. dnnu0.gt.3.d0) then
	    write(*,99) keyword
	    stop
	  endif
	elseif (line(1:3).eq.'TAU') then
	  keyword='TAU'
	  ikey=1
	  ierror=.false.
	  cflag='d'
	  call readline('TAU',valr,vall,valc,ierror)
	  if (ierror) then
	    write(*,99) keyword
	    stop
	  endif
	  tau0=valr
	  if (tau0.lt.875.2d0 .or. tau0.gt.885.2d0) then
	    write(*,99) keyword
	    stop
	  endif
	elseif (line(1:3).eq.'XIE') then
	  keyword='XIE'
	  ikey=1
	  ierror=.false.
	  cflag='d'
	  call readline('XIE',valr,vall,valc,ierror)
	  if (ierror) then
	    write(*,99) keyword
	    stop
	  endif
	  xie0=valr
	  if (xie0.lt.-1.d0 .or. xie0.gt.1.d0) then
	    write(*,99) keyword
	    stop
	  endif
	elseif (line(1:3).eq.'XIX') then
	  keyword='XIX'
	  ikey=1
	  ierror=.false.
	  cflag='d'
	  call readline('XIX',valr,vall,valc,ierror)
	  if (ierror) then
	    write(*,99) keyword
	    stop
	  endif
	  xix0=valr
	  if (xix0.lt.-1.d0 .or. xix0.gt.1.d0) then
	    write(*,99) keyword
	    stop
	  endif
	elseif (line(1:7).eq.'RHOLMBD') then
	  keyword='RHOLMBD'
	  ikey=1
	  ierror=.false.
	  cflag='d'
	  call readline('RHOLMBD',valr,vall,valc,ierror)
	  if (ierror) then
	    write(*,99) keyword
	    stop
	  endif
	  rholmbd0=valr
	  if (rholmbd0.lt.0.d0 .or. rholmbd0.gt.1.d0) then
	    write(*,99) keyword
	    stop
	  endif
	elseif (line(1:7).eq.'NETWORK') then
	  keyword='NETWORK'
	  ikey=1
	  ierror=.false.
	  cflag='d'
	  call readline('NETWORK',valr,vall,valc,ierror)
	  if (ierror) then
	    write(*,99) keyword
	    stop
	  endif
	  inuc=valr
	  if (inuc.ne.9 .and. inuc.ne.18 .and. inuc.ne.26) then
	    write(*,99) keyword
	    stop
	  endif
	  if (inuc.eq.9) irec=40
	  if (inuc.eq.18) irec=73
	  if (inuc.eq.26) irec=100
	elseif (line(1:5).eq.'RATES') then
	  keyword='RATES'
	  ikey=1
	  ierror=.false.
	  cflag='d'
	  call readline('RATES',valr,vall,valc,ierror)
	  if (ierror) then
	    write(*,99) keyword
	    stop
	  endif
	  dnchrat=valr
	  nchrat=nchrat+dnchrat
	  call readrates
	elseif (line(1:6).eq.'OUTPUT') then
	  keyword='OUTPUT'
	  ikey=1
	  ierror=.false.
	  cflag='l'
	  call readline('OUTPUT',valr,vall,valc,ierror)
	  if (ierror) then
	    write(*,99) keyword
	    stop
	  endif
	  if (vall) then
	    ixt(30)=1
	    ikey=2
	    ierror=.false.
	    cflag='d'
	    call readline('OUTPUT',valr,vall,valc,ierror)
	    if (ierror) then
	      write(*,99) keyword
	      stop
	    endif
	    nvxt=valr
	    cflag='d'
	    do i=1,nvxt
	      ikey=i+2
	      ierror=.false.
	      call readline('OUTPUT',valr,vall,valc,ierror)
	      if (ierror) then
	        write(*,99) keyword
	        stop
	      endif
	      ixt(i)=valr
	    enddo
	    do i=nvxt+1,nnuc
	      ixt(i)=0
	    enddo
	  else
	    ixt(30)=0
	  endif
	elseif (line(1:6).eq.'FOLLOW') then
	  keyword='FOLLOW'
	  ikey=1
	  ierror=.false.
	  cflag='l'
	  call readline('FOLLOW',valr,vall,valc,ierror)
	  if (ierror) then
	    write(*,99) keyword
	    stop
	  endif
	  follow=vall
	elseif (line(1:5).eq.'FILES') then
	  keyword='FILES'
	  ikey=1
	  ierror=.false.
	  cflag='c'
	  call readline('FILES',valr,vall,valc,ierror)
	  if (ierror) then
	    write(*,99) keyword
	    stop
	  endif
	  namefile1=valc
	  ikey=2
	  ierror=.false.
	  cflag='c'
	  call readline('FILES',valr,vall,valc,ierror)
	  if (ierror) then
	    write(*,99) keyword
	    stop
	  endif
	  namefile2=valc
	  ikey=3
	  ierror=.false.
	  cflag='c'
	  call readline('FILES',valr,vall,valc,ierror)
	  if (ierror) then
	    write(*,99) keyword
	    stop
	  endif
	  namefile3=valc
	elseif (line(1:9).eq.'OVERWRITE') then
	  keyword='OVERWRITE'
	  ikey=1
	  ierror=.false.
	  cflag='l'
	  call readline('OVERWRITE',valr,vall,valc,ierror)
	  if (ierror) then
	    write(*,99) keyword
	    stop
	  endif
	  overw=vall
	elseif (line(1:4).eq.'EXIT') then
	  goto 23
	endif
	goto 22
23	close(1)
	goto 701

C---------------------------Interactive mode---------------------------
C-----Set default parameters
3	isel=0
	goto 800

100	write(*,1000)
1000	format(////1x,
     .' PPPP     AAAA                   h       EEEEEE  N    N        ',
     .'   PPPP   EEEEEE'/1x,
     .'P    P   A    A            tt    h       E       NN   N        ',
     .'  P    P  E'/1x,
     .'P    P   A    A  r rrrr  tttttt  h       E       N N  N   oooo ',
     .'  P    P  E'/1x,
     .'PPPPP    AAAAAA  rr        tt    hhhhh   EEEEE   N  N N  o    o',
     .'  PPPPP   EEEEE'/1x,
     .'P        A    A  rr        tt    h    h  E       N   NN  o    o',
     .'  P       E'/1x,
     .'P        A    A  rr        tt    h    h  E       N    N  o    o',
     .'  P       E',/1x,
     .'P        A    A  rr        tttt  h    h  EEEEEE  N    N   oooo ',
     .'  P       EEEEEE'///
     .16x,'       Public Algorithm Evaluating the'/
     .16x,'    Nucleosynthesis of Primordial Elements'/
     .16x,'                Version 3.0'/
     .16x,'           (last update 26/1/21)'////
     .16x,'         Naples Astroparticle Group'/
     .16x,'     http://people.na.infn.it/~astropar/'//)
	write(*,*) ' Press RETURN to proceed'
	read(*,*)
200	write(*,2000)
2000	format(///////////////9x,
     .'PArthENoPE  computes the abundances  of light nuclides  produced'
     ./9x,
     .'during Big Bang Nucleosynthesis. Starting from nuclear statistic'
     ./9x,
     .'equilibrium  conditions, the program solves  the set  of coupled'
     ./9x,
     .'ordinary  differential  equations, follows  the  departure  from'
     ./9x,
     .'chemical  equilibrium of  nuclear  species, and determines their'
     ./9x,
     .'asymptotic abundances as function of several input  cosmological'
     ./9x,
     .'parameters  as  the  baryon  density,  the  number of  effective'
     ./9x,
     .'neutrinos, the  value of cosmological  constant and the neutrino'
     ./9x,
     .'chemical  potential.  Follow  the  instruction on  the screen to'
     ./9x,
     .'select  various input parameters and  nuclear network parameters'
     ./9x,
     .'or customize the output. For further information and help, visit'
     ./9x,
     .'the URL http://parthenope.na.infn.it/'////)
	write(*,*) ' Press RETURN to proceed'
	read(*,*)
300	write(*,3000)
3000	format(///////////////////
     .    1x,'1. Select physical parameters'//
     .    1x,'2. Select nuclear network parameters'//
     .    1x,'3. Customize output'//
     .    1x,'4. Run the nucleosynthesis code'//
     .    1x,'5. Quit program'//
     .    1x,'Please, enter selection (1-5):'/)
31	read(*,'(i3)') isel
	if (isel.lt.1 .or. isel.gt.5) then
	  write(*,999)
	  goto 31
	endif
	goto(400,500,600,700,4) isel
C-----Physical parameters
400	write(*,4000) eta10,dnnu0,tau0,xie0,xix0,rholmbd0
4000	format(///////////
     .    1x,'1. Change baryonic asymmetry from eta10 = ',
     .    f7.5//
     .    1x,'2. Change number of additional neutrino species from',
     .    '.........DN_nu = ',
     .    f7.3//
     .    1x,'3. Change value of neutron lifetime from',
     .    '.....................tau_N = ',
     .    f7.2//
     .    1x,'4. Change value of e-neutrino chemical potential',
     .    ' from...........xi_e = ',
     .    f7.2//
     .    1x,'5. Change value of x-neutrino chemical potential',
     .    ' from...........xi_x = ',
     .    f7.2//
     .    1x,'6. Change vacuum energy density at BBN epoch from',
     .    '..........Rholmbd = ',
     .    f7.3//
     .    1x,'7. Reset physical parameters to default values'//
     .    1x,'8. Run the nucleosynthesis code'//
     .    1x,'9. Go to previous menu'//
     .    1x,'Please, enter selection (1-9):'/)
41	read(*,'(i3)') isel
	if (isel.lt.1 .or. isel.gt.9) then
	  write(*,999)
	  goto 41
	endif
	goto(401,402,403,404,405,406,407,700,408) isel
401	write(*,4010)
4010	format(//' Please, enter the new value of eta10:'/)
	read(*,*) eta10
	if (eta10.lt.2.d0 .or. eta10.gt.9.d0) then
	  write(*,9999)
	  write(*,*) 'Please press RETURN to continue'
	  read(*,'()')
	endif
	goto 400
402	write(*,4020)
4020	format(//' Please, enter the new value of DN_nu:'/)
	read(*,*) dnnu0
	if (dnnu0.lt.-3.d0 .or. dnnu0.gt.3.d0) then
	  write(*,9999)
	  write(*,*) 'Please press RETURN to continue'
	  read(*,'()')
	endif
	goto 400
403	write(*,4030)
4030	format(//' Please, enter the new value of tau_N:'/)
	read(*,*) tau0
	if (tau0.lt.875.2d0 .or. tau0.gt.885.2d0) then
	  write(*,9999)
	  write(*,*) 'Please press RETURN to continue'
	  read(*,'()')
	endif
	goto 400
404	write(*,4040)
4040	format(//' Please, enter the value of xi_e:'/)
42	read(*,*) xie0
	if (xie0.lt.-1.d0 .or. xie0.gt.1.d0) then
	  write(*,999)
	  goto 42
	endif
	goto 400
405	write(*,4050)
4050	format(//' Please, enter the value of xi_x:'/)
43	read(*,*) xix0
	if (xix0.lt.-1.d0 .or. xix0.gt.1.d0) then
	  write(*,999)
	  goto 43
	endif
	goto 400
406	write(*,4060)
4060	format(//' Please, enter the new value of Rholmbd:'/)
	read(*,*) rholmbd0
	if (rholmbd0.lt.0.d0 .or. rholmbd0.gt.1.d0) then
	  write(*,9999)
	  write(*,*) 'Please press RETURN to continue'
	  read(*,'()')
	endif
	goto 400
C-----Deafault physical values
407	isel=100
	goto 800
408	goto 300
C-----Network parameters
500	write(*,5000) inuc,irec
5000	format(/////////////////
     .    1x,'1. Change the network from ',i2,' nuclides and ',
     .    i3,' reactions'//
     .    1x,'2. Change the nuclear rates'//
     .    1x,'3. Reset nuclear network parameters to default values'//
     .    1x,'4. Run the nucleosynthesis code'//
     .    1x,'5. Go to previous menu'//
     .    1x,'Please, enter selection (1-5):'/)
51	read(*,'(i3)') isel
	if (isel.lt.1 .or. isel.gt.5) then
	  write(*,999)
	  goto 51
	endif
	goto(501,502,503,700,504) isel
501	write(*,5010)
5010	format(/////////////////////
     .    1x,'1. 9 nuclides and 40 reactions'//
     .    1x,'2. 18 nuclides and 73 reactions'//
     .    1x,'3. 26 nuclides and 100 reactions'//
     .    1x,'4. Go to the previous menu'//
     .    1x,'Please, enter selection (1-4):'/)
52	read(*,'(i3)') isel
	if (isel.lt.1 .or. isel.gt.4) then
	  write(*,999)
	  goto 52
	endif
	goto(511,512,513,500) isel
511	inuc=9
	irec=40
	goto 500
512	inuc=18
	irec=73
	goto 500
513	inuc=26
	irec=100
	goto 500
C-----Changing rate parameters
502	write(*,'(///////////////////)')
	if (irec.eq.40)
     .    write(*,5030) ((i,rstring(i),i=j+1,j+4),j=0,36,4)
	if (irec.eq.73)
     .    write(*,5030) ((i,rstring(i),i=j+1,j+4),j=0,68,4),
     .    (i,rstring(i),i=73,73)
	if (irec.eq.100) then
	  write(*,5020) ((i,rstring(i),i=j+1,j+4),j=0,72,4)
5020	  format(19(1x,4(i3,'. ',a)/)/
     .      2x,'Please, enter the number corresponding to the ',
     .      'reaction that you want to change'/
     .      2x,'or press RETURN for the remaining reactions')
	  read(*,'(i3)') isel
	  if (isel.eq.0) then
	    write(*,*)
	    write(*,5020) ((i,rstring(i),i=j+1,j+4),j=76,92,4),
     .        (i,rstring(i),i=97,100)
	  elseif (isel.gt.0 .and. isel.le.irec) then
	    goto 54
	  else
	    write(*,999)
	    stop
	  endif
	endif
5030	format(19(1x,4(i3,'. ',a)/))
	write(*,5031)
5031	format(/2x,'Please, enter the number corresponding to the ',
     .    'reaction that you want to change'/
     .    2x,'or press RETURN for going back to previous menu'/)
53	read(*,'(i3)') isel
	if (isel.lt.0 .or. isel.gt.irec) then
	  write(*,999)
	  goto 53
	endif
54	if (isel.eq.0) goto 500
	if (nchrat.eq.0) then
	  nchrat=nchrat+1
	  ichrat=nchrat
	  wchrat(ichrat)=isel
	else
	  do i=1,nchrat
	    if (wchrat(i).eq.isel) then
	      ichrat=i
	      goto 521
	    endif
	  enddo
	  nchrat=nchrat+1
	  ichrat=nchrat
	  wchrat(ichrat)=isel
	endif
521	write(*,5032)
5032	format(//4x,'Please, choose the way you want to change it'//
     .    1x,'0. Select adopted rate'//
     .    1x,'1. Select low rate'//
     .    1x,'2. Select high rate'//
     .    1x,'3. Select (adopted rate)*factor'//
     .    3x,'Please, enter selection (0-3):'/)
55	read(*,'(i3)') hchrat(wchrat(ichrat))
	if (hchrat(wchrat(ichrat)).lt.0 .or. hchrat(wchrat(ichrat)).gt.3)
     .    then
	  write(*,999)
	  goto 55
	endif
	if (hchrat(wchrat(ichrat)).eq.3) then
	  write(*,5033)
5033	  format(/1x,'Please, enter your choice for the multiplicative ',
     .      'factor'/)
	  read(*,*) factor(wchrat(ichrat))
	  if (factor(wchrat(ichrat)).lt.0.d0 .or.
     .        factor(wchrat(ichrat)).gt.1.d5) then
	    write(*,9999)
	    write(*,*) 'Please press RETURN to continue'
	    read(*,'()')
	  endif
	endif
C-----Summary of nuclear rate changes
	write(*,5034) nchrat
5034	format(//////////////////////
     .    1x,' You have chosen to change',i3,' rates in the ',
     .    'following way'//
     .    2x,'react. #',3x,'changed rate',4x,'changing option',4x,
     .    'mult. factor'/)
	do i=1,nchrat
	  if (hchrat(wchrat(i)).eq.3) then
	    write(*,5035) wchrat(i),rstring(wchrat(i)),
     .        chopt(hchrat(wchrat(i))+1),factor(wchrat(i))
	  else
	    write(*,5035) wchrat(i),rstring(wchrat(i)),
     .        chopt(hchrat(wchrat(i))+1)
	  endif
5035	  format(4x,i3,6x,a14,6x,a7,8x,d11.5)
	enddo
	write(*,*)
	write(*,*) ' Press RETURN to proceed'
	read(*,*)
	goto 502
503	isel=200
	write(*,5040)
5040	format(/
     .    1x,'Nuclear network parameters resetted to default values'/)
	write(*,*) ' Press RETURN to proceed'
	read(*,*)
	goto 801
504	goto 300
C-----Output choice
600	write(*,6000)
6000	format(///////////////////////
     .    1x,'Do you want to follow the evolution on the screen? ',
     .    '(yes, no)'/)
	read(*,'(a1)') csel
	if (csel.eq.'y') then
	  follow=.true.
	elseif (csel.eq.'n') then
	  follow=.false.
	else
	  write(*,999)
	  goto 600
	endif
601	write(*,6001) (byy(ixt(i)+1),i=1,nvxt)
6001	format(/
     .    1x,'With this menu you can decide the name of the output ',
     .    'files and'/
     .    1x,'enter the list of nuclides whose abundancies will be ',
     .    'saved in'/
     .    1x,'the output file NUCLIDES.OUT.'//
     .    1x,'The presently selected nuclides are:'/
     .    1x,26a5)
	write(*,6002)
6002	format(//
     .    1x,'Please enter your choice:'//
     .    1x,'1. Change the default choice for the output file names',//
     .    1x,'2. Restore the default choice for the output file names'//
     .    1x,'3. Change the default choice for the nuclide list',//
     .    1x,'4. No nuclide to be saved'//
     .    1x,'5. Restore the default choice for the nuclide list'//
     .    1x,'6. Run the nucleosynthesis code'//
     .    1x,'7. Go to the previous menu'/)
61	read(*,'(i3)') isel
	if (isel.lt.0 .or. isel.gt.7) then
	  write(*,999)
	  goto 61
	endif
	goto (602,603,604,606,606,700,300) isel
602	write(*,'(//1x,2a/)') 'Please, enter the name of the ',
     .    'abundances output file'
	read(*,*) namefile1
	write(*,'(//1x,2a/)') 'Please, enter the name of the ',
     .    'nuclides output file'
	read(*,*) namefile2
	goto 601
603	isel=500
	goto 804
604	write(*,'(//////////////////////)')
605	write(*,6003) inuc,(i,byy(i+1),i+6,byy(i+7),i+12,byy(i+13),i+18,
     .    byy(i+19),i+24,byy(i+25),i=1,2)
6003	format(//
     .    1x,'The nuclides whose abundances you can save are from 1 to',
     .    i3,' in the list below:'/
     .    2(/1x,i2,'.',a5,4(8x,i2,'.',a5)))
	write(*,6004) (i,byy(i+1),i+6,byy(i+7),i+12,byy(i+13),i+18,
     .    byy(i+19),i=3,6)
6004	format(
     .    4(1x,i2,'.',a5,3(8x,i2,'.',a5)/))
	write(*,6005)
6005	format(/' Please, enter the total number of nuclides you want to',
     .    ' follow:'/)
	read(*,'(i3)') nvxt
	write(*,6006)
6006	format(//' Please, enter the integers corresponding to the ',
     .    'selected nuclides separated'/' by spaces:'/)
	read(*,*) (ixt(i),i=1,nvxt)
	do i=1,nvxt
	  if (ixt(i).gt.inuc) then
	    write(*,999)
	    goto 605
	  endif
	enddo
	do i=nvxt+1,inuc+1
	  ixt(i)=0
	enddo
	if (nvxt.gt.0) then
	  ixt(30)=1
	else
	  ixt(30)=0
	endif
	goto 601
606	if (isel.eq.4) then
	  nvxt=0
	  ixt(30)=0
	  goto 601
	elseif (isel.eq.5) then
	  isel=400
	  goto 802
	endif

C-----Summary of parameters
700	write(*,7000)
7000	format(/////////////////////
     .    1x,'1. Run the nucleosynthesis code with present ',
     .    'parameters'//
     .    1x,'2. Run the nucleosynthesis code with default ',
     .    'parameters'//
     .    1x,'3. Go to selection menu'//
     .    1x,'Please, enter selection (1-3):'/)
71	read(*,'(i3)') isel
	if (isel.lt.1 .or. isel.gt.3) then
	  write(*,999)
	  goto 71
	endif
	goto(701,702,300) isel
701	write(*,7010) eta10,inuc,dnnu0,irec,tau0,xie0,xix0,rholmbd0
7010	format(///////////////////
     .    1x,'Settings are:'//
     .    3x,'   eta10 = ',f7.5,20x,'N_nuc = ',i2/
     .    3x,'   DN_nu = ',f7.3,20x,'N_reac = ',i3/
     .    3x,'   tau_N = ',f7.2/
     .    3x,'    xi_e = ',f7.2/
     .    3x,'    xi_x = ',f7.2/
     .    3x,' Rholmbd = ',f7.3/
     .    24x,'rate #',4x,'changed rate',
     .    4x,'changing option',4x,'mult. factor')
	if (nchrat.ne.0) then
	  do i=1,nchrat
	    if (hchrat(wchrat(i)).eq.3) then
	      write(*,'(24x,i3,7x,a14,6x,a7,8x,d11.5)') wchrat(i),
     .          rstring(wchrat(i)),chopt(hchrat(wchrat(i))+1),
     .          factor(wchrat(i))
	    else
	      write(*,'(24x,i3,7x,a14,6x,a7,8x,d11.5)') wchrat(i),
     .          rstring(wchrat(i)),chopt(hchrat(wchrat(i))+1)
	    endif
	  enddo
	endif
	if (ixt(30).ne.0) then
	  write(*,7011) (byy(ixt(i)+1),i=1,nvxt)
7011	  format(//2x,'Abundances saved in NUCLIDES.OUT:',26a5)
	else
	  write(*,7012)
7012	  format(//2x,'No Abundances saved in NUCLIDES.OUT')
	endif
	if (cmode.eq.'c') then
	  write(*,'(///)')
	  goto 900
	else
	  write(*,'(/1x,2a/)') 'If ok press 1, otherwise press 2 for ',
     .      'going to the selection menu:'
	endif
72	read(*,'(i3)') isel
	if (isel.lt.0 .or. isel.gt.2) then
	  write(*,999)
	  goto 72
	endif
	goto(721,722) isel
721	write(*,7020)
7020	format(//1x,'Please, wait while initialization is completed'//)
	goto 900
722	goto 300
702	isel=300

C-----Default parameters
C-----Physical parameters
C-----eta10 = eta*10^10 = 273.279/(1-0.007125 Y_p)*ombh (for T0g=2.7255K)
C-----The value below corresponds to Omega_B h^2=0.02226 (Planck 2015) for Y_p=0.2467
800	eta10=6.09389d0
C-----dnnu0=# of extra neutrinos; dnnu0=0 for standard BBN
	dnnu0=0.d0
C-----tau0=value of neutron lifetime; experimental quoted value 
C	is 880.2+-1.0 s
	tau0=880.2d0
C-----xie0=chemical potential of nu_e
	xie0=0.d0
C-----xix0=chemical potential of nu_x
	xix0=0.d0
C-----Energy density of a cosmological constant
	rholmbd0=0.d0
	if (isel.eq.100) goto 400
C-----Network and output parameters
C-----inuc=# of nuclides in the network
C-----irec=# of reaction between nuclides in the network
801	inuc=9
	irec=40
C-----nvxt=# of nuclides whose evolution has to be followed
802	nvxt=9
C-----ixt=code number of the nuclides whose evolution has to be
C	followed (ixt(30)=control integer)
	do i=1,9
	  ixt(i)=i
	enddo
	do i=10,29
	  ixt(i)=0
	enddo
	ixt(30)=1
	if (isel.eq.400) goto 601
C-----Changing rate parameters
	nchrat=0
	do i=1,nrec
	  wchrat(i)=0
	  hchrat(i)=0
	  factor(i)=1.d0
	enddo
C-----Initialization of rexcluded
	do i=1,nrec
	  rexcluded(i)=0
	enddo
	do i=1,nrec
	  if (rexcluded(i).gt.0) then
	    nchrat=nchrat+1
	    wchrat(nchrat)=rexcluded(i)
	    hchrat(wchrat(nchrat))=3
	    factor(wchrat(nchrat))=0.d0
	  elseif (rexcluded(i).lt.0) then
	    nchrat=nchrat+1
	    wchrat(nchrat)=-rexcluded(i)
	    hchrat(wchrat(nchrat))=3
	    factor(wchrat(nchrat))=1.d0
	  endif
	enddo
C-----Option for followig the evolution on screen
	follow=.false.
C-----Output files
804	namefile1='parthenope3.0.out'
	namefile2='nuclides3.0.out'
	namefile3='info3.0.out'
	if (fstime) then
	  fstime=.false.
	  if (cmode.eq.'c') goto 21
	  goto 100
	endif
	if (isel.eq.200) goto 500
	if (isel.eq.300) goto 701
	if (isel.eq.500) goto 601

C-----BBN evolution
900	eflag=0
	etaf0=1.d-10*eta10
	call parthenope(etaf0,dnnu0,tau0,xie0,xix0,rholmbd0)
	write(*,*)
	if (eflag.eq.0) then
	  write(*,*) 'Run completed successfully'
	  goto 901
	elseif (eflag.eq.1) then
	  write(*,*) 'Run failed with the following error:'
	  write(*,*) emess
	  write(*,*)
	  write(*,*) 'Attempting a small perturbation of the ',
     .      'starting point'
	  eflag=0
	  dnnu0=dnnu0+.00000020d0
	  write(*,*)
	  call parthenope(etaf0,dnnu0,tau0,xie0,xix0,rholmbd0)
	elseif (eflag.eq.2) then
	  write(*,*) 'Run failed with the following error:'
	  write(*,*) emess
	  stop
	endif
	write(*,*)
	if (eflag.eq.0) then
	  write(*,*) 'Run completed successfully'
	else
	  write(*,*) 'Run failed with the following error:'
	  write(*,*) emess
	  write(*,*) 'Please, check the info file'
	endif
901	write(*,*)

4	stop
99	format(/1x,'Run failed with the following error:'/
     .    1x,'Invalid parameter value for the keyword ',a8/
     .    1x,'Please, consult the manual.'/)
999	format(/1x,'Error in the entered selection. Please, retype the ',
     .    'value.'/)
9999	format(/1x,'VALUE OUT OF MAXIMUM RANGE: THE CODE COULD NOT BE ',
     .    'RELIABLE.'//)
	end


	SUBROUTINE READLINE(KEYWORD,VALR,VALL,VALC,IERROR)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	This subroutine extracts the information from each line of the
C	  input card
C
C	Called by MAIN
C
C	keyword=text string indentifying the keyword
C	valr=real valued parameter
C	vall=logical valued parameter (.FALSE. or :TRUE.)
C	valc=character valued parameter
C	ierror=.TRUE. only in case of an error in some input paramter
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	IMPLICIT DOUBLE PRECISION (A-Z)
C--------------------------Local variables------------------------------
	INTEGER          I,IEND,LENGTH
	CHARACTER        FORM*8
C-----Parameters for the input card mode
	CHARACTER        KEYWORD*(*),VALC*(*)
	LOGICAL          VALL,IERROR
C--------------------------Common variables-----------------------------
	CHARACTER        CMODE*1
	LOGICAL          FOLLOW,OVERW
	COMMON/INPCARD/  FOLLOW,OVERW,CMODE

	INTEGER          ISTART,IKEY,DNCHRAT
	CHARACTER        CFLAG*1,LINE*500
	COMMON/READINP/  ISTART,IKEY,DNCHRAT,CFLAG,LINE
C-----------------------------------------------------------------------

	if (istart.le.0) istart=len(keyword)
	length=len(line)
	do i=istart+1,length
	  if (line(i:i).ne.' ') goto 1
	enddo
1	istart=i
	if (istart.gt.length) then
	  ierror=.true.
	  return
	endif
	do i=istart+1,length
	  if (line(i:i).eq.' ') goto 2
	enddo
2	if (line(i:i).eq.' ') then
	  iend=i-1
	else
	  iend=i
	endif
	if (cflag.eq.'d') then
	  if (iend-istart+1.lt.10) then
	    form='(f .0)'
	    write(form(3:3),'(i1)') iend-istart+1
	  else
	    form='(f  .0)'
	    write(form(3:4),'(i2)') iend-istart+1
	  endif
	  read(line(istart:iend),form) valr
	elseif (cflag.eq.'l') then
	  if (iend.eq.istart) then
	    if (line(istart:istart).eq.'T') then
	      vall=.true.
	    elseif (line(istart:istart).eq.'F') then
	      vall=.false.
	    else
	      write(*,9) ikey,keyword
	      stop
	    endif
	  else
	    write(*,9) ikey,keyword
	    stop
	  endif
	elseif (cflag.eq.'c') then
	  write(form,"('(a',i3,')')") iend-istart+1
	  read(line(istart:iend),form) valc
	endif
	istart=iend

	return
9	format(/1x,'Invalid parameter value for the argument ',i1,
     .    ' of the keyword ',a8/1x,'Please, consult the manual.'/)
	end


	SUBROUTINE READRATES
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	This subroutine reads the changes in the nuclear rates from the
C	  input card
C
C	Called by MAIN
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	IMPLICIT DOUBLE PRECISION (A-Z)
C--------------------------Local variables------------------------------
	INTEGER          I,J,IEND,ICHRAT,LENGTH
	CHARACTER        FORM*8
C-----Network parameters
	INTEGER          NREC
	PARAMETER        (NREC=100)
C-----Parameters for the input card mode
	CHARACTER        KEYWORD*8
C--------------------------Common variables-----------------------------
	INTEGER          WCHRAT(NREC),HCHRAT(NREC),NCHRAT
	DIMENSION        FACTOR(NREC)
	COMMON/CHRATE/   WCHRAT,HCHRAT,FACTOR,NCHRAT

	CHARACTER        CMODE*1
	LOGICAL          FOLLOW,OVERW
	COMMON/INPCARD/  FOLLOW,OVERW,CMODE

	INTEGER          ISTART,IKEY,DNCHRAT
	CHARACTER        CFLAG*1,LINE*500
	COMMON/READINP/  ISTART,IKEY,DNCHRAT,CFLAG,LINE
C-----------------------------------------------------------------------

	keyword='RATES'
	ichrat=nchrat-dnchrat
	length=len(line)

1	do i=istart+1,length
	  if (line(i:i).eq.'(') then
	    ichrat=ichrat+1
	    if (ichrat.gt.nchrat) then
	      write(*,99)
	      stop
	    endif
	    do j=i+1,length
	      if (line(j:j).ne.' ') goto 2
	    enddo
	  endif
	enddo
	if (ichrat.eq.nchrat) then
	  return
	else
	  write(*,99)
	  stop
	endif
2	istart=j
	do i=istart+1,length
	  if (line(i:i).eq.' ' .or. line(i:i).eq.')') goto 3
	enddo
3	if (line(i:i).eq.')') then
	  write(*,999) wchrat(ichrat)
	  stop
	endif
	if (line(i:i).eq.' ') then
	  iend=i-1
	else
	  iend=i
	endif
	if (iend-istart+1.lt.10) then
	  form='(i )'
	  write(form(3:3),'(i1)') iend-istart+1
	else
	  form='(i  )'
	  write(form(3:4),'(i2)') iend-istart+1
	endif
	read(line(istart:iend),form) wchrat(ichrat)
	istart=iend+1
	do i=istart,length
	  if (line(i:i).ne.' ') goto 4
	enddo
4	istart=i
	do i=istart+1,length
	  if (line(i:i).eq.' ' .or. line(i:i).eq.')') goto 5
	enddo
5	if (line(i:i).eq.')') then
	  write(*,999) wchrat(ichrat)
	  stop
	endif
	if (line(i:i).eq.' ') then
	  iend=i-1
	else
	  iend=i
	endif
	if (iend-istart+1.lt.10) then
	  form='(i )'
	  write(form(3:3),'(i1)') iend-istart+1
	else
	  form='(i  )'
	  write(form(3:4),'(i2)') iend-istart+1
	endif
	read(line(istart:iend),form) hchrat(wchrat(ichrat))
	istart=iend+1
	do i=istart,length
	  if (line(i:i).ne.' ') goto 6
	enddo
6	istart=i
	do i=istart+1,length
	  if (line(i:i).eq.' ' .or. line(i:i).eq.')') goto 7
	enddo
7	if (line(i:i).eq.' ' .or. line(i:i).eq.')') then
	  iend=i-1
	else
	  iend=i
	endif
	if (iend-istart+1.lt.10) then
	  form='(f .0)'
	  write(form(3:3),'(i1)') iend-istart+1
	else
	  form='(f  .0)'
	  write(form(3:4),'(i2)') iend-istart+1
	endif
	if (hchrat(wchrat(ichrat)).eq.3)
     .	read(line(istart:iend),form) factor(wchrat(ichrat))
	istart=iend+1
	goto 1

9	format(/1x,'Syntax error for the keyword RATES: the parenthesis',
     .    ' do not close.'/1x,'Please, consult the manual.'/)
99	format(/1x,'Syntax error for the keyword RATES: the number of ',
     .    'changed reactions'/1x,'do not correspond to the total.'/1x,
     .    'Please, consult the manual.'/)
999	format(/1x,'Syntax error for the keyword RATES: the number of ',
     .    'variables read inside'/1x,'parenthesis for reaction ',i3,
     .    ' is not 3.'/1x,'Please, consult the manual.'/)
	end

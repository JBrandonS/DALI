C=======================================================================
C
C  DESCRIPTION OF THE VARIABLES CONTAINED IN THE COMMONS
C  =====================================================================
C  (IN ALPHABETIC ORDER OF THE COMMON)
C
C
C ----------------------------/ANUM/-----------------------------------
C  AA(NNUC)         = Nuclide atomic numbers
C
C -----------------------------/CHRATE/---------------------------------
C  FACTOR(NREC)     = Multiplicative factor for the rate of reaction i-th
C  HCHRAT(NREC)     = Type of changes adopted for reaction i-th
C  NCHRAT           = Number of reactions to be changed
C  WCHRAT(NREC)     = Reactions to be changed
C
C -----------------------------/CONSTANTS/------------------------------
C  ALF              = Fine structure constant
C  COEF(4)          = Unity conversion factors
C  GN               = Newton constant in MeV^(-2)
C  ME               = Electron mass in MeV
C  MU               = Atomic mass unit in MeV
C  PI               = Greek Pi
C  QVAL             = Dimensionless Q value in the weak reactions ([MN-MP]/ME)
C
C -----------------------------/COUNTS/---------------------------------
C  IFCN             = Counter
C  IOUTEV           = Counter
C
C -----------------------------/DELTAZ/---------------------------------
C  DZ               = Stepsize of the independent variable in the 
C                     resolution of the nucleosynthesis equations
C  DZ0              = Initial value for DZ
C
C -----------------------------/DMASS/----------------------------------
C  DM(0:NNUC)       = Mass excesses in MeV
C
C -----------------------------/DMASSH/---------------------------------
C  DMH(NNUC)        = Dimensionless mass excesses (DM/ME)
C
C -----------------------------/ECHPOT/---------------------------------
C  PHI              = Dimensionless electron chemical potential
C
C -----------------------------/ERRHANDLE/------------------------------
C  EFLAG            = Flag for the completion of the run
C                     EFLAG=0 successfull run
C                     EFLAG=1 repeat run with a small perturbation
C                     EFLAG=2 interrupt running
C  EMESS            = Error message
C
C -----------------------------/GPART/----------------------------------
C  AG(NNUC,4)       = Nuclear partition function coefficients
C
C -----------------------------/INABUN/---------------------------------
C  YY0(NNUC+1)      = Initial values of electron chemical potential and
C                     nuclide abundances
C
C -----------------------------/INPCARD/--------------------------------
C  CMODE            = Flag for the choice of the running mode
C  FOLLOW           = Option for following the evolution on the screen
C                     (card mode)
C  OVERW            = Option for overwriting the output files (card mode)
C
C -----------------------------/INVFLAGS/-------------------------------
C  INC              = Maximum value of the flag for the convergence of
C                     the matrix inversion
C  MBAD             = Error flag for the matrix inversion
C
C -----------------------------/INVPHI/---------------------------------
C  LH0              = Initial value for the function Lh
C  Z0               = Initial value for the evolution variable z=me/T
C                     (=ZIN)
C
C -----------------------------/LINCOEF/--------------------------------
C  AMAT(NNUC,NNUC)  = Matrix involved in the linearization of the
C                     relation between Y(Z-DZ) and Y(Z)
C  BVEC(NNUC)       = Vector involved in the linearization of the
C                     relation between Y(Z-DZ) and Y(Z) (contains Y in 
C                     reverse order)
C  YX(NNUC)         = Y(Z) in reverse order
C
C -----------------------------/MINABUN/--------------------------------
C  YMIN             = Numerical zero of nuclide abundances
C
C -----------------------------/MODPAR/---------------------------------
C  DNNU             = Number of extra effective neutrinos
C  ETAF             = Final value of the baryon to photon density ratio
C  RHOLMBD          = Energy density corresponding to a cosmological
C                     constant
C  TAU              = Value of neutron lifetime in seconds
C  XIE              = Electron neutrino chemical potential (=XIE0)
C  XIX              = Muon/tau neutrino chemical potential (=XIX0)
C
C -----------------------------/NCHPOT/---------------------------------
C  IXIE             = Positive integer corresponding to the value on the
C                     grid of the electron neutrino chemical potential
C  XIEG0(NXIE)      = Grid values of the nu_e chemical potential
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
C  BYY(NNUC)        = Text strings for the output
C  CYY(NNUC)        = Text strings for the output
C
C -----------------------------/NUCMASS/--------------------------------
C  MN               = Neutron mass in MeV
C  MP               = Proton mass in MeV
C
C -----------------------------/OUTFILES/-------------------------------
C  NAMEFILE1        = Name of the output file with the final values of
C                     the nuclide abundances
C  NAMEFILE2        = Name of the output file with the evolution of the
C                     nuclides whose evolution has to be followed
C  NAMEFILE3        = Name of the output file with the information on the
C                     numerical integration
C
C -----------------------------/OUTPUTS/--------------------------------
C  NOUT             = Number of output variables in OUTVAR to be printed
C                     (<=20)
C  OUTTXT(1)        = Printed name of OUTVAR(1) (thetah)
C  OUTTXT(2)        = Printed name of OUTVAR(2) (tnuh)
C  OUTTXT(3)        = Printed name of OUTVAR(3) (nbh)
C  OUTTXT(4-20)     = Printed names of OUTVAR(4-20)
C  OUTVAR(1)        = Dimensionless Hubble function times 3 (thetah)
C  OUTVAR(2)        = Neutrino to photon temperature ratio (tnuh)
C  OUTVAR(3)        = Dimensionless baryon number density (nbh)
C  OUTVAR(4-20)     = Free locations for user-defined output variables
C
C -----------------------------/PREVVAL/--------------------------------
C  DZP              = Previous iteration value of the stepsize of the
C                     independent variable in the resolution of the
C                     nucleosynthesis equations
C  SUMMYP           = Value of the linear combination
C                     Sum((dMih+3/(2 z)) dYi/dz)
C  SUMZYP           = Value of the linear combination Sum(Zi dYi/dz)
C  ZP               = Previous iteration value of the evolution variable
C                     z=me/T
C
C -----------------------------/RECPAR/---------------------------------
C  IFORM(NREC)      = Reaction type (1-12)
C  NG(NREC)         = Number of incoming nuclides of type TG
C  NH(NREC)         = Number of incoming nuclides of type TH
C  NI(NREC)         = Number of incoming nuclides of type TI
C  NJ(NREC)         = Number of incoming nuclides of type TJ
C  NK(NREC)         = Number of incoming nuclides of type TK
C  NL(NREC)         = Number of incoming nuclides of type TL
C  Q9(NREC)         = Energy released in reaction (in unit of 10^9 K)
C  REV(NREC)        = Reverse reaction coefficient
C  TG(NREC)         = Incoming nuclide type
C  TH(NREC)         = Outgoing nuclide type
C  TI(NREC)         = Incoming nuclide type
C  TJ(NREC)         = Incoming nuclide type
C  TK(NREC)         = Outgoing nuclide type
C  TL(NREC)         = Outgoing nuclide type
C
C -----------------------------/RECPAR0/--------------------------------
C  RATEPAR(NREC,13) = Reaction parameter values (=IFORM+TI+...+NI+...)
C
C -----------------------------/RSTRINGS/-------------------------------
C  RSTRING(NREC)    = Reaction text strings
C
C -----------------------------/THERMQ/---------------------------------
C  LH               = Function LH
C  LHPHI            = Derivative of LH with respect to PHI
C  LHZ              = Derivative of LH with respect to Z
C  NAUX             = Neutrino auxiliary function, zero in the limit of
C                     thermal neutrino distribution functions
C  PBH              = Dimensionless baryon pressure
C  PEH              = Dimensionless electron pressure
C  PGH              = Dimensionless gamma pressure
C  RHOBH            = Dimensionless baryon energy density
C  RHOEH            = Dimensionless electron energy density
C  RHOEHPHI         = Derivative of RHOEH with respect to PHI
C  RHOEHZ           = Derivative of RHOEH with respect to Z
C  RHOGH            = Dimensionless gamma energy density
C  RHOGHZ           = Derivative of RHOGH with respect to Z
C  RHOH             = Dimensionless total energy density
C
C -----------------------------/SPINDF/---------------------------------
C  GNUC(NNUC)       = Nuclide spin degrees of freedom
C
C -----------------------------/WEAKRATE/-------------------------------
C  A(13)            = Forward weak reaction best-fit parameter (non
C                     degenerate case)
C  B(10)            = Reverse weak reaction best-fit parameter (non
C                     degenerate case)
C  DA(12,NXIE)      = Forward weak reaction best-fit parameter
C                     (degenerate case)
C  DB(12,NXIE)      = Reverse weak reaction best-fit parameter
C                     (degenerate case)
C  QNP              = Forward weak reaction best-fit exponent parameter
C  QNP1             = Forward weak reaction best-fit exponent parameter
C  QPN              = Reverse weak reaction best-fit exponent parameter
C  QPN1             = Reverse weak reaction best-fit exponent parameter
C
C -----------------------------/ZNUM/-----------------------------------
C  ZZ(NNUC)         = Nuclide atomic charges
C
C=======================================================================
C
C  DESCRIPTION OF THE VARIABLES CONTAINED IN THE COMMONS
C  =====================================================================
C  (IN ALPHABETIC ORDER OF THE VARIABLE NAMES)
C
C
C  A(13)            = Forward weak reaction best-fit parameter (non
C                     degenerate case)
C  AA(NNUC)         = Nuclide atomic numbers
C  AG(NNUC,4)       = Nuclear partition function coefficients
C  ALF              = Fine structure constant
C  AMAT(NNUC,NNUC)  = Matrix involved in the linearization of the
C                     relation between Y(Z-DZ) and Y(Z)
C  B(10)            = Reverse weak reaction best-fit parameter (non
C                     degenerate case)
C  BVEC(NNUC)       = Vector involved in the linearization of the
C                     relation between Y(Z-DZ) and Y(Z) (contains Y in 
C                     reverse order)
C  BYY(NNUC)        = Text strings for the output
C  CMODE            = Flag for the choice of the running mode
C  COEF(4)          = Unity conversion factors
C  CYY(NNUC+1)      = Text strings for the output
C  DA(12,NXIE)      = Forward weak reaction best-fit parameter
C                     (degenerate case)
C  DB(12,NXIE)      = Reverse weak reaction best-fit parameter
C                     (degenerate case)
C  DNNU             = Number of extra effective neutrinos
C  DZ               = Stepsize of the independent variable in the 
C                     resolution of the nucleosynthesis equations
C  DZ0              = Initial value for DZ
C  DZP              = Previous iteration value of the stepsize of the
C                     independent variable in the resolution of the
C                     nucleosynthesis equations
C  DM(0:NNUC)       = Mass excesses in MeV
C  DMH(NNUC)        = Dimensionless mass excesses (DM/ME)
C  EFLAG            = Flag for the completion of the run
C                     EFLAG=0 successfull run
C                     EFLAG=1 repeat run with a small perturbation
C                     EFLAG=2 interrupt running
C  EMESS            = Error message
C  ETAF             = Final value of the baryon to photon density ratio
C  FACTOR(NREC)     = Multiplicative factor for the rate of reaction i-th
C  FOLLOW           = Option for following the evolution on the screen
C                     (card mode)
C  GN               = Newton constant in MeV^(-2)
C  GNUC(NNUC)       = Nuclide spin degrees of freedom
C  HCHRAT(NREC)     = Type of changes adopted for reaction i-th
C  IFCN             = Counter
C  IFORM(NREC)      = Reaction type (1-12)
C  INC              = Maximum value of the flag for the convergence of
C                     the matrix inversion
C  INUC             = Number of nuclides in the selected network
C  IOUTEV           = Counter
C  IREC             = Number of reactions among nuclides in the selected
C                     network
C  IXIE             = Positive integer corresponding to the value on the
C                     grid of the electron neutrino chemical potential
C  IXT(30)          = Code of the nuclides whose evolution has to be
C                     followed (ixt(30)=control integer)
C  LH               = Function LH
C  LH0              = Initial value for the function Lh
C  LHPHI            = Derivative of LH with respect to PHI
C  LHZ              = Derivative of LH with respect to Z
C  MBAD             = Error flag for the matrix inversion
C  ME               = Electron mass in MeV
C  MN               = Neutron mass in MeV
C  MP               = Proton mass in MeV
C  MU               = Atomic mass unit in MeV
C  NAMEFILE1        = Name of the output file with the final values of
C                     the nuclide abundances
C  NAMEFILE2        = Name of the output file with the evolution of the
C                     nuclides whose evolution has to be followed
C  NAMEFILE3        = Name of the output file with the information on the
C                     numerical integration
C  NAUX             = Neutrino auxiliary function, zero in the limit of
C                     thermal neutrino distribution functions
C  NCHRAT           = Number of reactions to be changed
C  NG(NREC)         = Number of incoming nuclides of type TG
C  NH(NREC)         = Number of incoming nuclides of type TH
C  NI(NREC)         = Number of incoming nuclides of type TI
C  NJ(NREC)         = Number of incoming nuclides of type TJ
C  NK(NREC)         = Number of incoming nuclides of type TK
C  NL(NREC)         = Number of incoming nuclides of type TL
C  NOUT             = Number of output variables in OUTVAR to be printed
C                     (<=20)
C  NVXT             = Number of nuclides whose evolution has to be
C                     followed
C  OUTTXT(1)        = Printed name of OUTVAR(1) (thetah)
C  OUTTXT(2)        = Printed name of OUTVAR(2) (tnuh)
C  OUTTXT(3)        = Printed name of OUTVAR(3) (nbh)
C  OUTTXT(4-20)     = Printed names of OUTVAR(4-20)
C  OUTVAR(1)        = Dimensionless Hubble function times 3 (thetah)
C  OUTVAR(2)        = Neutrino to photon temperature ratio (tnuh)
C  OUTVAR(3)        = Dimensionless baryon number density (nbh)
C  OUTVAR(4-20)     = Free locations for user-defined output variables
C  OVERW            = Option for overwriting the output files (card mode)
C  PBH              = Dimensionless baryon pressure
C  PEH              = Dimensionless electron pressure
C  PGH              = Dimensionless gamma pressure
C  PHI              = Dimensionless electron chemical potential
C  PI               = Greek Pi
C  Q9(NREC)         = Energy released in reaction (in unit of 10^9 K)
C  QNP              = Forward weak reaction best-fit exponent parameter
C  QNP1             = Forward weak reaction best-fit exponent parameter
C  QPN              = Reverse weak reaction best-fit exponent parameter
C  QPN1             = Reverse weak reaction best-fit exponent parameter
C  QVAL             = Dimensionless Q value in the weak reactions ([MN-MP]/ME)
C  RATEPAR(NREC,13) = Reaction parameter values (=IFORM+TI+...+NI+...)
C  REV(NREC)        = Reverse reaction coefficient
C  RHOBH            = Dimensionless baryon energy density
C  RHOEH            = Dimensionless electron energy density
C  RHOEHPHI         = Derivative of RHOEH with respect to PHI
C  RHOEHZ           = Derivative of RHOEH with respect to Z
C  RHOGH            = Dimensionless gamma energy density
C  RHOGHZ           = Derivative of RHOGH with respect to Z
C  RHOH             = Dimensionless total energy density
C  RHOLMBD          = Energy density corresponding to a cosmological
C                     constant
C  RSTRING(NREC)    = Reaction text strings
C  SUMMYP           = Value of the linear combination
C                     Sum((dMih+3/(2 z)) dYi/dz)
C  SUMZYP           = Value of the linear combination Sum(Zi dYi/dz)
C  TAU              = Value of neutron lifetime in seconds
C  TG(NREC)         = Incoming nuclide type
C  TH(NREC)         = Outgoing nuclide type
C  TI(NREC)         = Incoming nuclide type
C  TJ(NREC)         = Incoming nuclide type
C  TK(NREC)         = Outgoing nuclide type
C  TL(NREC)         = Outgoing nuclide type
C  WCHRAT(NREC)     = Reactions to be changed
C  XIE              = Electron neutrino chemical potential (=XIE0)
C  XIEG0(NXIE)      = Grid values of the nu_e chemical potential
C  XIX              = Muon/tau neutrino chemical potential (=XIX0)
C  YMIN             = Numerical zero of nuclide abundances
C  YX(NNUC)         = Y(Z) in reverse order
C  YY0(NNUC+1)      = Initial values of electron chemical potential and
C                     nuclide abundances
C  Z0               = Initial value for the evolution variable z=me/T
C                     (=ZIN)
C  ZP               = Previous iteration value of the evolution variable
C                     z=me/T
C  ZZ(NNUC)         = Nuclide atomic charges
C
C=======================================================================
	SUBROUTINE PARTHENOPE(ETAF0,DNNU0,TAU0,XIE0,XIX0,RHOLMBD0)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	This subroutine drives the resolution of the BBN set of equations
C
C	Called by MAIN
C	Calls INIT, OUTEND, and DLSODE (integration routine)
C
C	etaf0=final value of the baryon to photon density ratio
C	dnnu0=number of extra effective neutrinos. The standard case
C	      corresponds to dnnu0=0
C	tau0=value of neutron lifetime in seconds
C	xie0=electron neutrino chemical potential
C	xix0=muon/tau neutrino chemical potential
C	rholmbd0=energy density corresponding to a cosmological constant
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	IMPLICIT DOUBLE PRECISION (A-Z)
C--------------------------Local variables------------------------------
	INTEGER          L
C-----Network parameters
	INTEGER          NNUC,NREC
	PARAMETER        (NNUC=26,NREC=100)
C-----Differential equation resolution parameters
	INTEGER          ITOL,ITASK,IOPT,IWORK,LIW,LRW,MF,NEQ,ISTATE
	PARAMETER        (LIW=20+NNUC+1,LRW=22+9*(NNUC+1)+(NNUC+1)**2)
	DIMENSION        RWORK(LRW),IWORK(20+NNUC+1),RTOL(NNUC+1),
     .                 ATOL(NNUC+1)
	DIMENSION        YY(NNUC+1)
	EXTERNAL         DLSODE,FCN
C-----Initialization
	EXTERNAL         INIT
C-----Print output
	EXTERNAL         OUTEND
C--------------------------Common variables-----------------------------
	DIMENSION        COEF(4)
	EQUIVALENCE      (ALP,COEF(1)),(BET,COEF(2)),(GAM,COEF(3)),
     .                 (DEL,COEF(4))
	COMMON/CONSTANTS/PI,COEF,ME,MU,QVAL,ALF,GN

	COMMON/DELTAZ/   DZ0,DZ

	INTEGER          EFLAG
	CHARACTER*50     EMESS
	COMMON/ERRHANDLE/EFLAG,EMESS

	DIMENSION        YY0(NNUC+1)
	COMMON/INABUN/   YY0

	COMMON/MINABUN/  YMIN

	INTEGER          INUC,IREC,NVXT,IXT(30)
	COMMON/NETWRK/   INUC,IREC,NVXT,IXT

	CHARACTER*500    NAMEFILE1,NAMEFILE2,NAMEFILE3
	COMMON/OUTFILES/ NAMEFILE1,NAMEFILE2,NAMEFILE3

	INTEGER          IGT
	COMMON/ZSAVE/    IGT
C-----------------------------------------------------------------------

C-----Initial and final values for the evolution variable z=me/T
	zin=me/10.
	zend=me*130.

C-----Initialization of the z grid for the output
	igt=1

C-----Initialization
	call init(etaf0,dnnu0,tau0,xie0,xix0,rholmbd0,zin,zend)
	if (eflag.ne.0) return

C-----Resolution of the coupled differential equations.
	z=zin
	do l=1,inuc+1
	  yy(l)=yy0(l)
	enddo
	itol=4
	do l=1,(nnuc+1)
	  rtol(l)=1.d-6
	  if (l.gt.10) rtol(l)=4.d-6
	enddo
	do l=1,(nnuc+1)
	  atol(l)=1.d-14
	  if (l.gt.10) atol(l)=1.d-28
	  if (l.eq.16) atol(l)=1.d-28
	enddo
C-----The first value of the array RWORK is the variable TCRIT required by DLSODE
C     as the point that should not be overshoot in the integration
	itask=4
	rwork(1)=zend
C-----Optional inputs used
	iopt=1
	do l=5,10
	  rwork(l)=0.0d0
	  iwork(l)=0
	enddo
	rwork(5)=1.d-2
	rwork(6)=1.d-3
	rwork(7)=1.d-30
	iwork(5)=12
	iwork(6)=50000000
	iwork(7)=10
C-----MF = 10*METH + MITER: 
C     METH=2 (backward differentiation formulas)
C     MITER=2 (chord iteration with an internally generated (difference quotient) full Jacobian)
	mf=22
	neq=inuc+1
	istate=1
	call dlsode(fcn,neq,yy,z,zend,itol,rtol,atol,itask,
     .    istate,iopt,rwork,lrw,iwork,liw,jac,mf)

	do l=1,inuc
	  if (yy(l+1).lt.ymin) yy(l+1)=ymin
	enddo

C-----Write details about resolution of differential equations
	if (istate.eq.2) then
	  call outend(zend,yy)
	  write(4,*)
	  write(4,9996) 'relative tolerance   = ',rtol(1)
	  write(4,9996) 'absolute tolerance   = ',atol(1)
	  write(4,9996) 'initial step         = ',dz0
	  write(4,9996) 'last step used       = ',rwork(11)
	  write(4,9996) 'next step to be used = ',rwork(12)
	  write(4,9996) 'z current            = ',rwork(13)
	  write(4,*)
	  write(4,9998) '# of steps           = ',iwork(11)
	  write(4,9998) '# of FCN calls       = ',iwork(12)
	  write(4,9998) '# of JAC calls       = ',iwork(13)
	  write(4,*)
	  write(4,9998) 'last meth ord        = ',iwork(14)
	  write(4,9998) 'next meth ord        = ',iwork(15)
	  write(4,*)
	  write(4,*) '------------------------------------------',
     .      '--------------------------'
	else
	  eflag=1
	  emess='Exit DLSODE with negative ISTATE'
	  write(4,*)
	  write(4,9999) 'Exit DLSODE with ISTATE = ',istate,
     .      '  and z = ',z
	endif
	close(2)
	close(3)
	close(4)

9996	format(2x,a,d11.4)
9997	format(2x,a,7x,a)
9998	format(2x,a,i8)
9999	format(2x,a,i2,a,d12.5)

	RETURN
	END


	SUBROUTINE INIT(ETAF0,DNNU0,TAU0,XIE0,XIX0,RHOLMBD0,ZIN,ZEND)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	This subroutine initializes nuclear parameters and computes the
C	  initial values for the nuclide abundances and electron chemical
C	  potential
C
C	Called by PARTHENOPE
C	Calls ZBRAK and ZBRENT
C
C	etaf0=final value of the baryon to photon density ratio
C	dnnu0=number of extra effective neutrinos. The standard case
C	      corresponds to dnnu0=0
C	tau0=value of neutron lifetime in seconds
C	xie0=electron neutrino chemical potential
C	xix0=muon/tau neutrino chemical potential
C	rholmbd0=energy density corresponding to a cosmological constant
C	zin=initial value of z
C	zend=final value of z
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	IMPLICIT DOUBLE PRECISION (A-Z)
C--------------------------Local variables------------------------------
	INTEGER          I
C-----Network parameters
	INTEGER          NNUC,NREC
	PARAMETER        (NNUC=26,NREC=100)
C-----Changing rate parameters
	CHARACTER*7      CHOPT(4)
	DATA             CHOPT/'ADOPTED','LOW','HIGH','FACTOR'/
C-----Reaction parameters
	INTEGER          II,JJ,GG,HH,KK,LL  !Equate to ti,tj,tg,th,tk,tl.
C-----Nuclear data
	DIMENSION        MASS(NNUC)
C-----Variables for inquiring existing files
	CHARACTER        CSEL*1
	LOGICAL          FEXIST
C-----Parameters for the inversion of the initial condition on the electron
C	chemical potential
	INTEGER          NB,NBMAX,NBRAC
	PARAMETER        (NB=100,NBMAX=20,RANGE=.1D0,EPS=1.D-14)
	DIMENSION        XB1(NBMAX),XB2(NBMAX)
	EXTERNAL         LHFUN,ZBRAK,ZBRENT
C-----Variable with the format of the output variables
	CHARACTER*60     FORM
C--------------------------Common variables-----------------------------
	INTEGER          AA(NNUC)
	COMMON/ANUM/     AA

	INTEGER          WCHRAT(NREC),HCHRAT(NREC),NCHRAT
	DIMENSION        FACTOR(NREC)
	COMMON/CHRATE/   WCHRAT,HCHRAT,FACTOR,NCHRAT

	DIMENSION        COEF(4)
	EQUIVALENCE      (ALP,COEF(1)),(BET,COEF(2)),(GAM,COEF(3)),
     .	             (DEL,COEF(4))
	COMMON/CONSTANTS/PI,COEF,ME,MU,QVAL,ALF,GN

	INTEGER          IFCN,IOUTEV
	COMMON/COUNTS/   IFCN,IOUTEV

	DIMENSION        DM(0:NNUC)
	COMMON/DMASS/    DM

	DIMENSION        DMH(NNUC)
	COMMON/DMASSH/   DMH

	INTEGER          EFLAG
	CHARACTER*50     EMESS
	COMMON/ERRHANDLE/EFLAG,EMESS

	DIMENSION        YY0(NNUC+1)
	COMMON/INABUN/   YY0

	CHARACTER        CMODE*1
	LOGICAL          FOLLOW,OVERW
	COMMON/INPCARD/  FOLLOW,OVERW,CMODE

	COMMON/INVPHI/   LH0,Z0

	COMMON/MINABUN/  YMIN

	COMMON/MODPAR/   TAU,DNNU,ETAF,RHOLMBD,XIE,XIX

	INTEGER          IXIE,NXIE
	PARAMETER        (NXIE=21)
	DIMENSION        XIEG0(NXIE)
	COMMON/NCHPOT/   XIEG0,IXIE

	INTEGER          INUC,IREC,NVXT,IXT(30)
	COMMON/NETWRK/   INUC,IREC,NVXT,IXT

	CHARACTER*5      BYY(NNUC)
	CHARACTER*5      CYY(NNUC)
	COMMON/NSYMB/    BYY,CYY

	COMMON/NUCMASS/  MN,MP

	CHARACTER*500    NAMEFILE1,NAMEFILE2,NAMEFILE3
	COMMON/OUTFILES/ NAMEFILE1,NAMEFILE2,NAMEFILE3

	INTEGER          NOUT
	DIMENSION        OUTVAR(20)
	CHARACTER*7      OUTTXT(20)
	COMMON/OUTPUTS/  OUTVAR,OUTTXT,NOUT

	CHARACTER*12     DPG,DDN,DDP
	COMMON/RATES/    DPG,DDN,DDP

	INTEGER          IFORM(NREC),
     .                 TI(NREC),TJ(NREC),TG(NREC),
     .                 TH(NREC),TK(NREC),TL(NREC),
     .                 NI(NREC),NJ(NREC),NG(NREC),
     .                 NH(NREC),NK(NREC),NL(NREC)
	DIMENSION        Q9(NREC),REV(NREC)
	COMMON/RECPAR/   REV,Q9,IFORM,TI,TJ,TG,TH,TK,TL,NI,NJ,NG,NH,NK,NL

	INTEGER          RATEPAR(NREC,13)
	COMMON/RECPAR0/  RATEPAR

	CHARACTER*14     RSTRING(NREC)
	COMMON/RSTRINGS/ RSTRING

	DIMENSION        GNUC(NNUC)
	COMMON/SPINDF/   GNUC

	INTEGER          ZZ(NNUC)
	COMMON/ZNUM/     ZZ
C-----------------------------------------------------------------------

C-----Physical parameters
	etaf=etaf0
	dnnu=dnnu0
	tau=tau0
	xie=xie0
	xix=xix0
	if (xie.ge.0.d0) then
	  ixie = idint(xie/0.1d0)+11
	else
	  ixie = idint(xie/0.1d0)+10
	endif
	rholmbd=rholmbd0

C-----Open data files (only nuclides and info, because parthenope is a
C	grid file that can be already existent)
	if (cmode.eq.'i') then
	  if (ixt(30).ne.0) then
	    inquire(file=namefile2,exist=fexist)
	    if (fexist) then
10	      write(*,*)
	      write(*,*) 'ATTENTION! File nuclides.out exists. Overwrite',
     .          ' existing file? (yes/no)'
	      write(*,*)
	      read(*,*) csel
	      if (csel.eq.'y') then
	        open(unit=3,file=namefile2,status='replace')
	      elseif (csel.eq.'n') then
	        write(*,*)
	        write(*,1001)
1001	        format(1x,'Please, rename or move the file and then ',
     .            'restart the program')
	        write(*,*)
	        stop
	      else
	        goto 10
	      endif
	    else
	      open(unit=3,file=namefile2,status='new')
	    endif
	  endif
	  inquire(file=namefile3,exist=fexist)
	  if (fexist) then
20	    write(*,*)
	    write(*,*) 'ATTENTION! File info.out exists. Overwrite',
     .        ' existing file? (yes/no)'
	    write(*,*)
	    read(*,*) csel
	    if (csel.eq.'y') then
	      open(unit=4,file=namefile3,status='replace')
	    elseif (csel.eq.'n') then
	      write(*,*)
	      write(*,2001)
2001	      format(1x,'Please, rename or move the file and then ',
     .          'restart the program')
	      write(*,*)
	      stop
	    else
	      goto 20
	    endif
	  else
	    open(unit=4,file=namefile3,status='new')
	  endif
	elseif (cmode.eq.'c') then
	  if (ixt(30).ne.0) then
	    if (overw) then
	      open(unit=3,file=namefile2,status='replace')
	    else
	      inquire(file=namefile2,exist=fexist)
	      if (fexist) then
	        write(*,'(1x,2a)') 'ATTENTION! File nuclides.out exists',
     .            ', but the OVERWRITE flag in the input card'
	        write(*,'(1x,a)') 'is FALSE.'
	        write(*,'(1x,a//)') 'Please, consult the manual.'
	        stop
	      else
	        open(unit=3,file=namefile2,status='new')
	      endif
	    endif
	  endif
	  if (overw) then
	    open(unit=4,file=namefile3,status='replace')
	  else
	    inquire(file=namefile3,exist=fexist)
	    if (fexist) then
	      write(*,'(1x,2a)') 'ATTENTION! File info.out exists',
     .          ', but the OVERWRITE flag in the input card'
	      write(*,'(1x,a)') 'is FALSE.'
	      write(*,'(1x,a//)') 'Please, consult the manual.'
	      stop
	    else
	      open(unit=4,file=namefile3,status='new')
	    endif
	  endif
	else
	  write(*,*) 'Error! Neither the interactive nor the card mode',
     .      ' were selected'
	  stop
	endif

C-----Write initial data into output files: parthenope, nuclides and info files
	nout=3
	if (nout.gt.20) then
C-----nout has to be smaller than the dimension of outtxt
	  write(*,*) 'error in the value of nout'
	  stop
	endif
	outtxt(1)='thetah '
	outtxt(2)='  tnuh '
	outtxt(3)='  nbh  '
	if (nout.lt.10) then
	  form='(5x,a,9x,a,9x, (a7,8x),30(a5,10x))'
	  write(form(15:15),'(i1)') nout
	else
	  form='(5x,a,9x,a,9x,  (a7,8x),30(a5,10x))'
	  write(form(15:16),'(i2)') nout
	endif
	inquire(file=namefile1,exist=fexist)
	if (.not.fexist) then
	  open(unit=2,file=namefile1,status='new')
	  write(2,2002) 'N_nu',' xie',' xix','tau','rholmbd','eta10',
     .      'OmegaBh^2',' phie',(outtxt(i),i=1,nout),
     .      (cyy(ixt(i)),i=1,nvxt)
2002	  format(8(3x,a),8x,30(a7,7x))
	else
	  open(unit=2,file=namefile1,access='append')
	endif
	if (ixt(30).ne.0) then
	  write(3,form) 'T(MeV)','phi_e',(outtxt(i),i=1,nout),
     .      (byy(ixt(i)),i=1,nvxt)
	endif
	write(4,'(12x,2(a,f12.3))') 'T_i(MeV) =',me/zin,
     .    '    T_f(MeV) =',me/zend
	write(4,'(17x,2(a,d12.5))') 'z_i =',zin,'         z_f =',
     .    zend
	write(4,'(16x,2(a,f12.2))') 'T9_i =',5.929862032115561/zin,
     .    '        T9_f =',5.929862032115561/zend
	write(4,*)
	if (nchrat.ne.0) then
	  write(4,2003) nchrat
2003	  format(1x,' You have chosen to change',i4,' rates in the ',
     .      'following way'//
     .      2x,'changed rate',4x,'changing option',4x,'mult. factor')
	  write(4,*)
	  do i=1,nchrat
	    if (hchrat(wchrat(i)).eq.3) then
	      write(4,2004) rstring(wchrat(i)),chopt(hchrat(wchrat(i))+
     .          1),factor(wchrat(i))
	    else
	      write(4,2004) rstring(wchrat(i)),chopt(hchrat(wchrat(i))+1)
	    endif
2004	    format(2x,a14,6x,a7,8x,d11.5)
	  enddo
	endif
	write(4,*)
	write(4,'(23x,a)') 'Model parameters'
	write(4,'(4x,a,f6.2,4x,a,f5.2,2(3x,a,f5.2))')
     .    'tau = ',tau,'DN_nu = ',dnnu,'xi_e = ',xie,'xi_x = ',xix

C-----Choice of dpg, ddn and ddp rates
	inquire(file="rates.dat",exist=fexist)
	if (fexist) then
	  open(unit=20,file="rates.dat",status='old')
	  read(20,*) dpg
	  read(20,*) ddn
	  read(20,*) ddp
	  close(20)
	else
	  dpg="PIS2020"
	  ddn="PIS2020"
	  ddp="PIS2020"
	endif

C-----Calculation of the masses and dimensionless mass excesses. We use
C	the electron mass in MeV because the mass excesses are given in MeV
	do i=1,inuc
        mass(i)=aa(i)*mu+dm(i)
	  dmh(i)=dm(i)/me
	enddo

	do i=1,irec
C-----Read in reaction parameters
	  iform(i) = ratepar(i,1)
	  ti(i)    = ratepar(i,2)
	  tj(i)    = ratepar(i,3)
	  tg(i)	   = ratepar(i,4)
	  th(i)	   = ratepar(i,5)
	  tk(i)    = ratepar(i,6)
	  tl(i)    = ratepar(i,7)
	  ni(i)    = ratepar(i,8)
	  nj(i)    = ratepar(i,9)
	  ng(i)	   = ratepar(i,10)
	  nh(i)	   = ratepar(i,11)
	  nk(i)    = ratepar(i,12)
	  nl(i)    = ratepar(i,13)
C-----Energy released in reaction (in unit of 10^9 K)
	  q9(i)=(ni(i)*dm(ti(i))+nj(i)*dm(tj(i))+ng(i)*dm(tg(i))-
     .      nh(i)*dm(th(i))-nk(i)*dm(tk(i))-nl(i)*dm(tl(i)))/del
C-----Calculation of reverse reaction coefficient.
	  ii = ti(i)
	  jj = tj(i)
	  gg = tg(i)
	  hh = th(i)
	  kk = tk(i)
	  ll = tl(i)
	  goto(101,102,103,104,105,106,107,108,109,110,111,112) iform(i)
101	  continue                          !1-0--0-1 configuration.
	  rev(i)=0.d0
	  goto 113
102	  continue                          !1-1--0-1 configuration.
	  rev(i)=gnuc(ii)*gnuc(jj)/gnuc(ll)*
     .      (mass(ii)*mass(jj)/((mass(ii)+mass(jj))*mu))**1.5d0
	  goto 113
103	  continue                          !1-1-1-1 configuration.
	  rev(i)=gnuc(ii)*gnuc(jj)/(gnuc(kk)*gnuc(ll))*
     .      (mass(ii)*mass(jj)*(mass(kk)+mass(ll))/
     .      (mass(kk)*mass(ll)*(mass(ii)+mass(jj))))**1.5d0
	  goto 113
104	  continue                          !1-0--0-2 configuration.
	  rev(i)=0.d0
	  goto 113
105	  continue                          !1-1--0-2 configuration.
	  rev(i)=gnuc(ii)*gnuc(jj)*2/(gnuc(ll)*gnuc(ll))*
     .      (mass(ii)*mass(jj)*2*mass(ll)/
     .      (mass(ll)*mass(ll)*(mass(ii)+mass(jj))))**1.5d0
	  goto 113
106	  continue                          !2-0--1-1 configuration.
	  rev(i)=gnuc(ii)*gnuc(ii)/(2*gnuc(kk)*gnuc(ll))*
     .      (mass(ii)*mass(ii)*(mass(kk)+mass(ll))/
     .      (mass(kk)*mass(ll)*2*mass(ii)))**1.5d0
	  goto 113
107	  continue                          !3-0--0-1 configuration.
	  rev(i)=gnuc(ii)*gnuc(ii)*gnuc(ii)/(6*gnuc(ll))*
     .      (mass(ii)**3/(mu*mu*mass(ll)))**1.5d0
	  goto 113
108	  continue                          !2-1--0-1 configuration.
	  rev(i)=gnuc(ii)*gnuc(ii)*gnuc(jj)/(2*gnuc(ll))*
     .      (mass(jj)*mass(ii)**2/(mu*mu*mass(ll)))**1.5d0
	  goto 113
109	  continue                          !1-1--1-2 configuration.
	  rev(i)=gnuc(ii)*gnuc(jj)*2/(gnuc(kk)*gnuc(ll)**2)*
     .      (mass(ii)*mass(jj)*mu/(mass(kk)*mass(ll)**2))**1.5d0
	  goto 113
110	  continue                          !1-1--0-3 configuration.
	  rev(i)=gnuc(ii)*gnuc(jj)*6/gnuc(ll)**3*
     .      (mass(ii)*mass(jj)*mu/mass(ll)**3)**1.5d0
	  goto 113
111	  continue                          !2-0--2-1 configuration.
	  rev(i)=gnuc(ii)**2/(gnuc(kk)**2*gnuc(ll))*
     .      (mu*mass(ii)**2/(mass(kk)**2*mass(ll)))**1.5d0
	  goto 113
112	  continue                          !1-1--1-1-1 configuration.
	  rev(i)=gnuc(ii)*gnuc(jj)/(gnuc(hh)*gnuc(kk)*gnuc(ll))*
     .      (mu*mass(ii)*mass(jj)/(mass(hh)*mass(kk)*mass(ll)))**1.5d0
113	  continue
	enddo

C-----Calculation of the initial abundances. Note that yy(1) corresponds to
C	the electron chemical potential and the remaining yy(i) to the nuclide
C	abundances
C-----Dimensionless mass difference between neutron and proton
	qval=(mn-mp)/me
	yy0(2)=1./(1.d0+ex(qval*zin+xie))
	yy0(3)=1./(1.d0+ex(-qval*zin-xie))
C-----etinc is the multiplicative factor for deriving the initial value of eta 
C	including all effects (neutrino reheating and radiative corrections),
C	see sec. 4.2.2 in Serpico et al. JCAP 0412, 010 (2004).
	etinc=2.73d0
	yy0(4)=1.033283084601794d-4*yy0(2)*yy0(3)*etinc*etaf/
     .    zin**1.5d0*ex(4.354215433329128*zin)
	do i=5,inuc+1
	  yy0(i) = ymin
	enddo
C-----Initial values of the combination sumzy=Sum(Zi*Yi).
	sumzy0=0.d0
	do i=2,inuc+1
	  sumzy0=sumzy0+zz(i-1)*yy0(i)
	enddo
C-----yy0(1)=phi_e0 is the solution of the equation:
C		Lh(z_r0,phi_e0)=Lh0
C	with Lh0=2 zeta(3) sumzy0/pi^2 2.75 etaf
	z0=zin
	lh0=0.6698660552846518*(etinc/2.75)*sumzy0*etaf
	phi=1.d-8
	nbrac=nbmax
	call zbrak(lhfun,phi-range,phi+range,nb,xb1,xb2,nbrac)
	tol=eps*dabs(xb1(1)+xb2(1))/2.d0
	phi=zbrent(lhfun,xb1(1),xb2(1),tol)
	if (dabs(lhfun(phi)).gt.1.d-16) then
	  write(*,*) 'low precision in phi0 calculation'
	  write(*,*) 'final results could be affected'
	endif
	yy0(1)=phi

C-----Inizialization of the counters
	ifcn=0
	ioutev=0

	write(*,40)
40	format(//1x,'Running...'//)

9998	format(2x,a,i8)

	RETURN
	END


	DOUBLE PRECISION FUNCTION LHFUN(PPHI)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	This subroutine evaluates the dimensionless electron/positron
C	  asymmetry minus its input initial value as function of the
C	  electron chemical potential. Equating this difference to zero
C	  gives the implicit condition on the initial electron chemical
C	  potential
C
C	Called by ZBRAK and ZBRENT
C	Calls QTRAP
C
C	pphi=electron chemical potential
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	IMPLICIT DOUBLE PRECISION (A-Z)
C--------------------------Local variables------------------------------
	INTEGER          IER
	EXTERNAL         QTRAP,INTLHFUN
C--------------------------Common variables-----------------------------
	DIMENSION        COEF(4)
	EQUIVALENCE      (ALP,COEF(1)),(BET,COEF(2)),(GAM,COEF(3)),
     .	             (DEL,COEF(4))
	COMMON/CONSTANTS/PI,COEF,ME,MU,QVAL,ALF,GN

	COMMON/ECHPOT/   PHI

	INTEGER          EFLAG
	CHARACTER*50     EMESS
	COMMON/ERRHANDLE/EFLAG,EMESS

	COMMON/INVPHI/   LH0,Z0

	CHARACTER*500    NAMEFILE1,NAMEFILE2,NAMEFILE3
	COMMON/OUTFILES/ NAMEFILE1,NAMEFILE2,NAMEFILE3
C-----------------------------------------------------------------------

	phi=pphi

C-----Effective shift of evolution parameter due to radiative QED corrections
	zr=z0+pi*alf/(3.*z0)

C-----Precision of integration
	epsrel=1.d-6
	ier=0
C-----Integration until 25 should be sufficient
	call qtrap(intlhfun,zr,25.d0,res,epsrel,ier)
	if (ier.ne.0) then
	  write(*,*) 'too many steps in qtrap'
	  write(4,998) 'Exit QTRAP with IER = ',ier
	  eflag=1
	  emess='too many steps in QTRAP'
	endif
	lhfun=res/pi**2 - lh0

998	format(2x,a,i2,a,d12.5)

	RETURN
	END


	DOUBLE PRECISION FUNCTION INTLHFUN(X)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	This subroutine computes the function to be integrated to get the
C	  dimensionless electron/positron asymmetry
C
C	Called by QTRAP
C
C	x=integration variable
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	IMPLICIT DOUBLE PRECISION (A-Z)
C--------------------------Common variables-----------------------------
	DIMENSION        COEF(4)
	EQUIVALENCE      (ALP,COEF(1)),(BET,COEF(2)),(GAM,COEF(3)),
     .	             (DEL,COEF(4))
	COMMON/CONSTANTS/PI,COEF,ME,MU,QVAL,ALF,GN

	COMMON/ECHPOT/   PHI

	COMMON/INVPHI/   LH0,Z0
C-----------------------------------------------------------------------

C-----Effective shift of evolution parameter due to radiative QED corrections
	zr=z0+pi*alf/(3.*z0)

	intlhfun=x*dsqrt(dabs(x**2-zr**2))*(fermi(x-phi)-fermi(x+phi))

	RETURN
	END


	DOUBLE PRECISION FUNCTION FERMI(Y)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	Fermi-Dirac distribution function
C
C	Called by INTLHFUN
C
C	y=argument
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	IMPLICIT DOUBLE PRECISION (A-Z)

	if (y.gt.80.d0) then
	  fermi=0.d0
	else
	  if (y.lt.-80.d0) then
	    fermi=1.d0
	  else
	    fermi=1.d0/(dexp(y)+1.d0)
	  endif
	endif

	RETURN
	END


	SUBROUTINE FCN(NEQ,Z,YY,YYPRIME)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	This subroutine evaluates the right hand side of the BBN
C	  differential equations
C
C	Called by DLSODE (integration routine)
C	Calls OUTEVOL, THERMO, RATE, and EQSLIN
C
C	neq=number of unknown functions to be determined
C	z=evolution variable z=me/T
C	yy=array of unknown functions
C	yyprime=right hand side of differential equation system
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	IMPLICIT DOUBLE PRECISION (A-Z)
C--------------------------Local variables------------------------------
	INTEGER          I,J,N,I1,J1
C-----Network parameters
	INTEGER          NNUC,NREC
	PARAMETER        (NNUC=26,NREC=100)
C-----Reaction parameters
	INTEGER          II,JJ,GG,HH,KK,LL  !Equate to ti,tj,tg,th,tk,tl.
	INTEGER          RI,RJ,RG,RH,RK,RL  !Equate to ni,nj,ng,nh,nk,nl.
	DIMENSION        F(NREC),R(NREC)
	INTEGER          FACT               !Factorial function
	EXTERNAL         RATE
C-----Differential equation resolution parameters
	INTEGER          NEQ
	INTEGER          ISIZE1             !Equals inuc+1.
	DIMENSION        YY(NEQ),YYPRIME(NEQ),Y(0:NNUC),YEVOL(NNUC)
C-----Matrix inversion
	INTEGER          ICONV              !Convergence monitor.
	INTEGER          IERROR             !Element which does not converge.
	EXTERNAL         EQSLIN
C-----Thermodynamical quantities calculation
	EXTERNAL         THERMO
C-----Output parameters
	PARAMETER        (DELTA=.003117D0,TGFIN=1.D0/130.D0)
	EXTERNAL         OUTEVOL
C--------------------------Common variables-----------------------------
	DIMENSION        COEF(4)
	EQUIVALENCE      (ALP,COEF(1)),(BET,COEF(2)),(GAM,COEF(3)),
     .	             (DEL,COEF(4))
	COMMON/CONSTANTS/PI,COEF,ME,MU,QVAL,ALF,GN

	INTEGER          IFCN,IOUTEV
	COMMON/COUNTS/   IFCN,IOUTEV

	COMMON/DELTAZ/   DZ0,DZ

	DIMENSION        DMH(NNUC)
	COMMON/DMASSH/   DMH

	INTEGER          EFLAG
	CHARACTER*50     EMESS
	COMMON/ERRHANDLE/EFLAG,EMESS

	INTEGER          MBAD
	INTEGER          INC
	COMMON/INVFLAGS/ MBAD,INC

	DIMENSION        AMAT(NNUC,NNUC)
	DIMENSION        BVEC(NNUC)
	DIMENSION        YX(NNUC)
	COMMON/LINCOEF/  AMAT,BVEC,YX

	COMMON/MODPAR/   TAU,DNNU,ETAF,RHOLMBD,XIE,XIX

	COMMON/MINABUN/  YMIN

	INTEGER          INUC,IREC,NVXT,IXT(30)
	COMMON/NETWRK/   INUC,IREC,NVXT,IXT

	CHARACTER*500    NAMEFILE1,NAMEFILE2,NAMEFILE3
	COMMON/OUTFILES/ NAMEFILE1,NAMEFILE2,NAMEFILE3

	INTEGER          NOUT
	DIMENSION        OUTVAR(20)
	CHARACTER*7      OUTTXT(20)
	COMMON/OUTPUTS/  OUTVAR,OUTTXT,NOUT

	COMMON/PREVVAL/  ZP,DZP,SUMZYP,SUMMYP

	INTEGER          IFORM(NREC),
     .                 TI(NREC),TJ(NREC),TG(NREC),
     .                 TH(NREC),TK(NREC),TL(NREC),
     .                 NI(NREC),NJ(NREC),NG(NREC),
     .                 NH(NREC),NK(NREC),NL(NREC)
	DIMENSION        Q9(NREC),REV(NREC)
	COMMON/RECPAR/   REV,Q9,IFORM,TI,TJ,TG,TH,TK,TL,NI,NJ,NG,NH,NK,NL

	COMMON/THERMQ/   RHOGH,PGH,RHOGHZ,RHOEH,PEH,RHOEHZ,RHOEHPHI,RHOBH,
     .                 PBH,RHOH,LH,LHZ,LHPHI,NAUX

	INTEGER          ZZ(NNUC)
	COMMON/ZNUM/     ZZ

	INTEGER          IGT
	COMMON/ZSAVE/    IGT
C-----------------------------------------------------------------------

C-----Check of nuclide abundances: values smaller than ymin are set equal
C	to ymin. Two linear combinations used in the algorithm are also 
C	builded:
C	  sumy=Sum(Yi)
C	  sumzy=Sum(Zi*Yi)
	sumy=0.d0
	sumzy=0.d0
	do i=1,inuc
	  if (yy(i+1).lt.ymin) yy(i+1)=ymin
	  y(i)=yy(i+1)
	  sumy=sumy+yy(i+1)
	  sumzy=sumzy+zz(i)*yy(i+1)
	enddo

C-----Counter increment
	ifcn=ifcn+1

C-----Compute the value of the stepsize dz
	if (ifcn.eq.1) then
	  dz=dz0
	  dzp=dz
	else
	  dz=dabs(z-zp)
	  if (dz.eq.0.d0) then
	    dz=dzp
	  else
	    dzp=dz
	  endif
	endif
	zp=z

C-----Calculation of the thermodynamical quantities
	call thermo(neq,z,yy,sumy,sumzy)

C-----Baryon number density
	nbh=lh/z**3/sumzy
	outvar(3)=nbh

C-----thetah=dr/dt=3*Hh (dimensionless Hubble function)
	thetah=dsqrt(24.*pi*gn*((me/z)**4*rhoh+rholmbd))/me
	outvar(1)=thetah

C-----Calculation of the time step dt corresponding to dz
	kap1p=4.*(rhoeh+rhogh)+1.5*pbh-z*(rhoehz+rhoghz)-z**2*lh/sumzy*
     .    summyp
	kap2=z*lhz-3.*lh-z*lh/sumzy*sumzyp
	kap3=dabs(rhogh+rhoeh)+pgh+peh+pbh+naux/3.d0
	den=lh*rhoehphi - lhphi*kap3
	if (ifcn.eq.1) then
	  dt=dabs(1./alp*2.*.738/me**2*z*dz)
	else
	  dt=dabs(-1./alp*(lhphi*kap1p+rhoehphi*kap2)/
     .      (me*z*thetah*den)*dz)
	endif

C-----Initialize the right hand side of the differential equations
	do i=2,inuc+1
	  yyprime(i)=0.d0
	enddo

C-----Evaluation of forward and reverse reaction rates
C-----Ratio n_B/N_A in mol/cm^3. It is equal to nb*Mu=(me/z)^3*Mu*Lh/sumzy.
	nbn=bet**3/gam*(me/z)**3*mu*lh/sumzy
	do i=1,irec
	  f(i)  = 0.d0
	  r(i)  = 0.d0
	enddo
C-----Notice that the first variable in this call is T9 = 5.929862032115561/z
	call rate(5.929862032115561/z,nbn,f,r)
c!!che si deve fare qui? bloccare il codice?
	if (eflag.ne.0) return

C-----Linearization of the differential equation system
	isize1=inuc+1
	y(0)=0.d0
C-----Initialize the A matrix to zero
	do i = 1,inuc
	  do j = 1,inuc
	    amat(i,j) = 0.d0
	  enddo
	enddo
	do n=1,irec
	  ii = ti(n)
	  jj = tj(n)
	  gg = tg(n)
	  hh = th(n)
	  kk = tk(n)
	  ll = tl(n)
	  if ((iform(n).ne.0).and.(ii.le.inuc).and.(ll.le.inuc)) then
	    ri = ni(n)
	    rj = nj(n)
	    rg = ng(n)
	    rh = nh(n)
	    rk = nk(n)
	    rl = nl(n)
C-----Compute different reaction rates
	    ci=ri*y(ii)**(ri-1)*y(jj)**rj*y(gg)**rg*f(n)/
     .        dble((ri+rj+rg)*fact(ri)*fact(rj)*fact(rg))
	    if (rj.lt.1) then
	      cj=0.d0
	    else
	      cj=rj*y(jj)**(rj-1)*y(ii)**ri*y(gg)**rg*f(n)/
     .          dble((ri+rj+rg)*fact(ri)*fact(rj)*fact(rg))
	    endif
	    if (rg.lt.1) then
	      cg=0.d0
	    else
	      cg=rg*y(gg)**(rg-1)*y(jj)**rj*y(ii)**ri*f(n)/
     .          dble((ri+rj+rg)*fact(ri)*fact(rj)*fact(rg))
	    endif
	    if (rh.lt.1) then
	      ch=0.d0
	    else
	      ch=rh*y(hh)**(rh-1)*y(kk)**rk*y(ll)**rl*r(n)/
     .          dble((rl+rk+rh)*fact(rl)*fact(rk)*fact(rh))
	    endif
	    if (rk.lt.1) then
	      ck=0.d0
	    else
	      ck=rk*y(kk)**(rk-1)*y(ll)**rl*y(hh)**rh*r(n)/
     .          dble((rl+rk+rh)*fact(rl)*fact(rk)*fact(rh))
	    endif
	    cl=rl*y(ll)**(rl-1)*y(kk)**rk*y(hh)**rh*r(n)/
     .        dble((rl+rk+rh)*fact(rl)*fact(rk)*fact(rh))
C-----Construct the A matrix
          ii = isize1 - ii                !Invert ii index.
          jj = isize1 - jj                !Invert jj index.
          gg = isize1 - gg                !Invert gg index.
          hh = isize1 - hh                !Invert hh index.
          kk = isize1 - kk                !Invert kk index.
          ll = isize1 - ll                !Invert ll index.
C-----Fill I nuclide column
          amat(ii,ii) = amat(ii,ii) +  ri*ci
          if (jj.le.inuc) amat(jj,ii) = amat(jj,ii) +  rj*ci
          if (gg.le.inuc) amat(gg,ii) = amat(gg,ii) +  rg*ci
          if (hh.le.inuc) amat(hh,ii) = amat(hh,ii) -  rh*ci
          if (kk.le.inuc) amat(kk,ii) = amat(kk,ii) -  rk*ci
          amat(ll,ii) = amat(ll,ii) -  rl*ci
C-----Fill J nuclide column
          if (jj.le.inuc) then
            amat(ii,jj) = amat(ii,jj) +  ri*cj
            amat(jj,jj) = amat(jj,jj) +  rj*cj
            if (gg.le.inuc) amat(gg,jj) = amat(gg,jj) +  rg*cj
            if (hh.le.inuc) amat(hh,jj) = amat(hh,jj) -  rh*cj
            if (kk.le.inuc) amat(kk,jj) = amat(kk,jj) -  rk*cj
            amat(ll,jj) = amat(ll,jj) -  rl*cj
          endif
C-----Fill G nuclide column
          if (gg.le.inuc) then
            amat(ii,gg) = amat(ii,gg) +  ri*cg
            if (jj.le.inuc) amat(jj,gg) = amat(jj,gg) +  rj*cg
            amat(gg,gg) = amat(gg,gg) +  rg*cg
            if (hh.le.inuc) amat(hh,gg) = amat(hh,gg) -  rh*cg
            if (kk.le.inuc) amat(kk,gg) = amat(kk,gg) -  rk*cg
            amat(ll,gg) = amat(ll,gg) -  rl*cg
          endif
C-----Fill H nuclide column
          if (hh.le.inuc) then
            amat(ii,hh) = amat(ii,hh) -  ri*ch
            if (jj.le.inuc) amat(jj,hh) = amat(jj,hh) -  rj*ch
            if (gg.le.inuc) amat(gg,hh) = amat(gg,hh) -  rg*ch
            amat(hh,hh) = amat(hh,hh) +  rh*ch
            if (kk.le.inuc) amat(kk,hh) = amat(kk,hh) +  rk*ch
            amat(ll,hh) = amat(ll,hh) +  rl*ch
          endif
C-----Fill K nuclide column
          if (kk.le.inuc) then
            amat(ii,kk) = amat(ii,kk) -  ri*ck
            if (jj.le.inuc) amat(jj,kk) = amat(jj,kk) -  rj*ck
            if (gg.le.inuc) amat(gg,kk) = amat(gg,kk) -  rg*ck
            if (hh.le.inuc) amat(hh,kk) = amat(hh,kk) +  rh*ck
            amat(kk,kk) = amat(kk,kk) +  rk*ck
            amat(ll,kk) = amat(ll,kk) +  rl*ck
          endif
C-----Fill L nuclide column
          amat(ii,ll) = amat(ii,ll) -  ri*cl
          if (jj.le.inuc) amat(jj,ll) = amat(jj,ll) -  rj*cl
          if (gg.le.inuc) amat(gg,ll) = amat(gg,ll) -  rg*cl
          if (hh.le.inuc) amat(hh,ll) = amat(hh,ll) +  rh*cl
          if (kk.le.inuc) amat(kk,ll) = amat(kk,ll) +  rk*cl
          amat(ll,ll) = amat(ll,ll) +  rl*cl
        endif
      enddo
C-----Put the A matrix in the final form
      bdln = 1.e-5*alp*me*thetah          !(10**(-5))*(Expansion rate).
      do i = 1,inuc
        i1 = isize1 - i                   !Invert the rows.
        do j = 1,inuc
          j1 = isize1 - j                 !Invert the columns.
          if (dabs(amat(j,i)).lt.bdln*y(j1)/y(i1)) then
            amat(j,i) = 0.d0              !Set 0 if tiny.
          else
            amat(j,i) = amat(j,i)*dt      !Bring dt over to other side.
          endif
        enddo
        amat(i,i) = 1.d0 + amat(i,i)      !Add the identity matrix to the A matrix.
        bvec(i1)  = y(i)                  !Initial abundances.
      enddo

C-----Solve equations to get derivatives
	mbad=0
	inc=30
	iconv=30
	call eqslin(iconv,ierror)
	if (mbad.ne.0) then
	  write(4,*) 'error in matrix inversion'
	  eflag=1
	  emess='error in matrix inversion'
c	  stop
	endif

C-----Derivatives of abundances
	do i=1,inuc
	  yevol(i)=yx(isize1-i)             !Abundance at z+dz.
	  yyprime(i+1)=(yevol(i)-y(i))/dz   !Take derivative.
	enddo

C-----Compute the two linear combinations:
C	  sumzyp=Sum(Zi dYi/dz)
C	  summyp=Sum((dMih+3/(2 z)) dYi/dz)
	sumzyp=0.d0
	summyp=0.d0
	do i=2,inuc+1
	  sumzyp=sumzyp+zz(i-1)*yyprime(i)
	  summyp=summyp+(dmh(i-1)+1.5d0/z)*yyprime(i)
	enddo

C-----Update the coefficients kap1p and kap2 with the new sumzyp and summyp
	kap1p=4.*(rhoeh+rhogh)+1.5*pbh-z*(rhoehz+rhoghz)-z**2*lh/sumzy*
     .    summyp
	kap2=z*lhz-3.*lh-z*lh/sumzy*sumzyp

C-----Derivative of yy(1)=Phi_e with respect to z
	yyprime(1)=1./z*(lh*kap1p+kap3*kap2)/den

C-----Store the current values of nuclide abundances in the output files
	tgsave=tgfin*10.d0**(delta*dfloat(1000-igt))
	if (me/z.le.tgsave) then
	  call outevol(z,yy)
	  do i=1,1000
	    tgsave=tgfin*10.d0**(delta*dfloat(1000-i))
	    if (tgsave.lt.me/z) goto 10
	  enddo
10	  igt=i
	endif

	RETURN
	END


	SUBROUTINE THERMO(NEQ,Z,YY,SUMY,SUMZY)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	This subroutine computes several thermodynamical quantities, as
C	  energy density and pressure for all the species, auxiliary
C	  functions and their derivatives
C
C	Called by FCN
C	Calls BESSEL
C
C	neq=number of unknown functions to be determined
C	z=evolution variable z=me/T
C	yy=array of unknown functions
C	sumy,sumzy=two linear combinations of abundances
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	IMPLICIT DOUBLE PRECISION (A-Z)
C--------------------------Local variables------------------------------
	INTEGER          I
C-----Network parameters
	INTEGER          NNUC
	PARAMETER        (NNUC=26)
C-----Differential equation resolution parameters
	INTEGER          NEQ
	DIMENSION        YY(NEQ)
C-----Bessel functions routine
	DIMENSION        KZ1(4),KZ2(4),KZ3(4),KZ4(4),KZ5(4),KZ6(4),KZ7(4)
	EXTERNAL         BESSEL
C--------------------------Common variables-----------------------------
	DIMENSION        COEF(4)
	EQUIVALENCE      (ALP,COEF(1)),(BET,COEF(2)),(GAM,COEF(3)),
     .	             (DEL,COEF(4))
	COMMON/CONSTANTS/PI,COEF,ME,MU,QVAL,ALF,GN

	DIMENSION        DMH(NNUC)
	COMMON/DMASSH/   DMH

	COMMON/MODPAR/   TAU,DNNU,ETAF,RHOLMBD,XIE,XIX

	INTEGER          INUC,IREC,NVXT,IXT(30)
	COMMON/NETWRK/   INUC,IREC,NVXT,IXT

	CHARACTER*500    NAMEFILE1,NAMEFILE2,NAMEFILE3
	COMMON/OUTFILES/ NAMEFILE1,NAMEFILE2,NAMEFILE3

	INTEGER          NOUT
	DIMENSION        OUTVAR(20)
	CHARACTER*7      OUTTXT(20)
	COMMON/OUTPUTS/  OUTVAR,OUTTXT,NOUT

	COMMON/THERMQ/   RHOGH,PGH,RHOGHZ,RHOEH,PEH,RHOEHZ,RHOEHPHI,RHOBH,
     .                 PBH,RHOH,LH,LHZ,LHPHI,NAUX
C-----------------------------------------------------------------------

	phie=yy(1)

C-------------------------------Photons---------------------------------
C-----Compute photon energy density, pressure and energy density derivative
	if (z.le.7.5d0) then
	  rhogh=.657337d0+4.26576d-5*z+4.02437d-4*z**2-3.57854d-4*z**3+
     .      1.77052d-4*z**4-5.79322d-5*z**5+1.2715d-5*z**6-
     .      1.82827d-6*z**7+1.63909d-7*z**8-8.26902d-9*z**9+
     .      1.78621d-10*z**10
	  pgh=.21872d0+3.96192d-5*z+3.72027d-4*z**2-3.27708d-4*z**3+
     .      1.61574d-4*z**4-5.28937d-5*z**5+1.16328d-5*z**6-
     .      1.67639d-6*z**7+1.50581d-7*z**8-7.60826d-9*z**9+
     .      1.64543d-10*z**10
	  rhoghz=4.26576d-5+8.04874d-4*z-1.073562d-3*z**2+
     .      7.08208d-4*z**3-2.89661d-4*z**4+7.629d-5*z**5-
     .      1.279789d-5*z**6+1.311272d-6*z**7-7.442118d-8*z**8+
     .      1.78621d-9*z**9
	else
	  rhogh=.657974d0
	  pgh=.219325d0
	  rhoghz=0.d0
	endif
	if ((z.le.10.d0).and.(z.ge..08d0)) then
	  rhoplh=-ex(-7.03116d0+1.32628*z-2.33731*z**2+1.49747*z**3-
     .      .558708*z**4+.12124*z**5-1.50395d-2*z**6+9.86051d-4*z**7-
     .      2.64207d-5*z**8)
	  pplh=-ex(-8.64555d0+6.02926*z-9.47217*z**2+6.81317*z**3-
     .      2.71151*z**4+.617877*z**5-7.9943d-2*z**6+5.44069d-3*z**7-
     .      1.50676d-4*z**8)
	  rhogh=rhogh+rhoplh
	  pgh=pgh+pplh
	  rhoghz=rhoghz+rhoplh*(1.32628d0-4.67462*z+4.49241*z**2-
     .      2.234832*z**3+.6062*z**4-9.0237d-2*z**5+6.902357d-3*z**6-
     .      2.113656d-4*z**7)
	endif
C-----------------------------------------------------------------------

C-----Compute hyperbolic functions
	if (phie.le.100.d0) then
	  cosh1=dcosh(phie)
	  cosh2=dcosh(2.*phie)
	  cosh3=dcosh(3.*phie)
	  cosh4=dcosh(4.*phie)
	  cosh5=dcosh(5.*phie)
	  cosh6=dcosh(6.*phie)
	  cosh7=dcosh(7.*phie)
	  sinh1=dsinh(phie)
	  sinh2=dsinh(2.*phie)
	  sinh3=dsinh(3.*phie)
	  sinh4=dsinh(4.*phie)
	  sinh5=dsinh(5.*phie)
	  sinh6=dsinh(6.*phie)
	  sinh7=dsinh(7.*phie)
	else
	  cosh1=0.d0
	  cosh2=0.d0
	  cosh3=0.d0
	  cosh4=0.d0
	  cosh5=0.d0
	  cosh6=0.d0
	  cosh7=0.d0
	  sinh1=0.d0
	  sinh2=0.d0
	  sinh3=0.d0
	  sinh4=0.d0
	  sinh5=0.d0
	  sinh6=0.d0
	  sinh7=0.d0
	endif
C-----Effective shift of evolution parameter due to radiative QED corrections
	zr=z+pi*alf/(3.*z)
	dzr=1.d0-pi*alf/(3.*z**2)
C-----Bessel function arguments
	zr1=zr
	zr2=2.*zr
	zr3=3.*zr
	zr4=4.*zr
	zr5=5.*zr
	zr6=6.*zr
	zr7=7.*zr
C-----Bessel functions Ki (i=1,4) calculated at n zr (n=1,7):
	do i=1,4
	  kz1(i)=bessel(i,zr1)
	  kz2(i)=bessel(i,zr2)
	  kz3(i)=bessel(i,zr3)
	  kz4(i)=bessel(i,zr4)
	  kz5(i)=bessel(i,zr5)
	  kz6(i)=bessel(i,zr6)
	  kz7(i)=bessel(i,zr7)
	enddo
C-----Calculation of Lh, and its derivatives Lhz e Lhphi
	lh=2.*zr**2/pi**2*(
     .    kz1(2)*sinh1-kz2(2)*sinh2/2.+kz3(2)*sinh3/3.-
     .    kz4(2)*sinh4/4.+kz5(2)*sinh5/5.-kz6(2)*sinh6/6.+
     .    kz7(2)*sinh7/7.
     .			)
	lhz=dzr*(2.*lh/zr - (zr/pi)**2*(
     .    (kz1(1)+kz1(3))*sinh1-(kz2(1)+kz2(3))*sinh2+
     .    (kz3(1)+kz3(3))*sinh3-(kz4(1)+kz4(3))*sinh4+
     .    (kz5(1)+kz5(3))*sinh5-(kz6(1)+kz6(3))*sinh6+
     .    (kz7(1)+kz7(3))*sinh7))
	lhphi=2.*(zr/pi)**2*(
     .    kz1(2)*cosh1-kz2(2)*cosh2+kz3(2)*cosh3-kz4(2)*cosh4+
     .    kz5(2)*cosh5-kz6(2)*cosh6+kz7(2)*cosh7)
C----------------------------Electrons----------------------------------
C-----Compute electron energy density, pressure and energy density derivatives
	rhoeh=zr/2.*(zr/pi)**2*(
     .    (3.*kz1(3)+kz1(1))*cosh1-(3.*kz2(3)+kz2(1))*cosh2/2.+
     .    (3.*kz3(3)+kz3(1))*cosh3/3.-(3.*kz4(3)+kz4(1))*cosh4/4.+
     .    (3.*kz5(3)+kz5(1))*cosh5/5.-(3.*kz6(3)+kz6(1))*cosh6/6.+
     .    (3.*kz7(3)+kz7(1))*cosh7/7.)
	peh=2.*(zr/pi)**2*(
     .    kz1(2)*cosh1-kz2(2)*cosh2/2.**2+kz3(2)*cosh3/3.**2-
     .    kz4(2)*cosh4/4.**2+kz5(2)*cosh5/5.**2-kz6(2)*cosh6/6.**2+
     .    kz7(2)*cosh7/7.**2)
	rhoehz=dzr*(3.*rhoeh/zr + zr/2.*(zr/pi)**2*(
     .    ((15.*kz1(3)+kz1(1))/zr-4.*kz1(4))*cosh1-
     .    ((15.*kz2(3)+kz2(1))/(2.*zr)-4.*kz2(4))*cosh2+
     .    ((15.*kz3(3)+kz3(1))/(3.*zr)-4.*kz3(4))*cosh3-
     .    ((15.*kz4(3)+kz4(1))/(4.*zr)-4.*kz4(4))*cosh4+
     .    ((15.*kz5(3)+kz5(1))/(5.*zr)-4.*kz5(4))*cosh5-
     .    ((15.*kz6(3)+kz6(1))/(6.*zr)-4.*kz6(4))*cosh6+
     .    ((15.*kz7(3)+kz7(1))/(7.*zr)-4.*kz7(4))*cosh7))
	rhoehphi=zr/2.*(zr/pi)**2*(
     .    (3.*kz1(3)+kz1(1))*sinh1-(3.*kz2(3)+kz2(1))*sinh2+
     .    (3.*kz3(3)+kz3(1))*sinh3-(3.*kz4(3)+kz4(1))*sinh4+
     .    (3.*kz5(3)+kz5(1))*sinh5-(3.*kz6(3)+kz6(1))*sinh6+
     .    (3.*kz7(3)+kz7(1))*sinh7)
C-----------------------------------------------------------------------

C------------------------------Neutrinos--------------------------------
C-----Compute neutrino energy density and the auxiliary function naux.
C	naux vanishes in the limit of thermal neutrino distribution functions
	naux=ex(-10.21703221236002d0 + 61.24438067531452d0*z-
     .    340.3323864212157d0*z**2 +1057.2707914654834d0*z**3-
     .    2045.577491331372d0*z**4 + 2605.9087171012848d0*z**5-
     .    2266.1521815470196d0*z**6 + 1374.2623075963388d0*z**7-
     .    586.0618273295763d0*z**8 +174.87532902234145d0*z**9-
     .    35.715878215468045d0*z**10 + 4.7538967685808755d0*z**11-
     .    0.3713438862054167d0*z**12 +0.012908416591272199d0*z**13)
	if (z.ge.4.d0) naux=0.d0
	rhonh=(3.0045813575784174d-7-0.00001643531199791781d0*z+
     .    0.00032347583551915635d0*z**2 -0.002511496397747168d0*z**3+ 
     .    (3.000917448281724d0 + 0.45426040174181076d0*ex(z))*z**4+
     .    1.3638973525933868d0*z**5 + 0.2518035560770054d0*z**6+
     .    0.10604388182309477d0*z**7 -0.019177667823809492d0*z**8+
     .    0.002304271482331396d0*z**9-0.00010434419908956676d0*z**10)/
     .    ((1.d0+1.d0*ex(z))*z**4)
C-----Additional D.O.F. and/or contribution from neutrino chemical potential. We
C	dinstinguish between electron and muon/tau neutrinos.
C-----We use instantaneous decoupling at Td=2.3 MeV, i.e. zxd=m_e/2.3
	zxd=me/2.3
C-----Coefficients used in the calculation
	co2=1.52323d0
	co3=rhogh+pgh
	co4=rhoeh+peh
	if (z.le.zxd) then
	  tnuh=1.d0
	else
	  tnuh=((co3+co4)/(co3+co2))**(1./3.d0)
	endif
	outvar(2)=tnuh
	dnnuxi=15.d0/7.d0*(2.d0*(xie/pi)**2+(xie/pi)**4+
     .    4.d0*(xix/pi)**2+2.d0*(xix/pi)**4)
	rhonh=rhonh+(7./8.d0)*2.*(dnnu+dnnuxi)*(pi**2/30.)*tnuh**4
C-----------------------------------------------------------------------

C------------------------------Baryons----------------------------------
C-----Compute baryon energy density and pressure
	rhobh=mu/me
	do i=2,inuc+1
	  rhobh=rhobh+(dmh(i-1)+1.5/z)*yy(i)
	enddo
	rhobh=z*lh/sumzy*rhobh
	pbh=lh/sumzy*sumy
C-----------------------------------------------------------------------

C--------------------------------rhoh-----------------------------------
C-----Total energy density
	rhoh=dabs(rhobh+rhonh+rhogh+rhoeh)
C-----------------------------------------------------------------------

	RETURN
	END


	SUBROUTINE RATE(T9,NBN,F,R)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	This subroutine evaluates forward and reverse rates
C
C	Called by FCN
C	Calls WRATE
C
C	t9=temperature in 10^9K unit
C	f,r=forward and reverse rate arrays
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	IMPLICIT DOUBLE PRECISION (A-Z)
C--------------------------Local variables------------------------------
	INTEGER          I
C-----Network parameters
	INTEGER          NNUC,NREC
	PARAMETER        (NNUC=26,NREC=100)
C-----Reaction parameters
	DIMENSION        F(NREC),R(NREC)
	DIMENSION        DRATE(NREC,2)		!Rate uncertainties
C-----Nuclear data
	DIMENSION        GNPF(0:NNUC)
C--------------------------Common variables-----------------------------
	INTEGER          WCHRAT(NREC),HCHRAT(NREC),NCHRAT
	DIMENSION        FACTOR(NREC)
	COMMON/CHRATE/   WCHRAT,HCHRAT,FACTOR,NCHRAT

	DIMENSION        COEF(4)
	EQUIVALENCE      (ALP,COEF(1)),(BET,COEF(2)),(GAM,COEF(3)),
     .	             (DEL,COEF(4))
	COMMON/CONSTANTS/PI,COEF,ME,MU,QVAL,ALF,GN

	DIMENSION        AG(NNUC,4)
	COMMON/GPART/    AG

	INTEGER          INUC,IREC,NVXT,IXT(30)
	COMMON/NETWRK/   INUC,IREC,NVXT,IXT

	INTEGER          NOUT
	DIMENSION        OUTVAR(20)
	CHARACTER*7      OUTTXT(20)
	COMMON/OUTPUTS/  OUTVAR,OUTTXT,NOUT

	CHARACTER*12     DPG,DDN,DDP
	COMMON/RATES/    DPG,DDN,DDP

	INTEGER          IFORM(NREC),
     .                 TI(NREC),TJ(NREC),TG(NREC),
     .                 TH(NREC),TK(NREC),TL(NREC),
     .                 NI(NREC),NJ(NREC),NG(NREC),
     .                 NH(NREC),NK(NREC),NL(NREC)
	DIMENSION        Q9(NREC),REV(NREC)
	COMMON/RECPAR/   REV,Q9,IFORM,TI,TJ,TG,TH,TK,TL,NI,NJ,NG,NH,NK,NL
C-----------------------------------------------------------------------

C-----Inizialization of the rate error, drate
	if (nchrat.ne.0) then
	  do i=1,irec
	    drate(i,1)=-.9d0
	    drate(i,2)=9.d0
	  enddo
	endif

C-----Compute weak reaction rates for standard neutrinos
	z=5.929862032115561/t9
	call wrate(z,f(1),r(1))
	if (eflag.ne.0) return

C-----Set decay rate coefficients

C.......H3 -> e- + v + He3.........(Tilly-Weller-Hasan 1987)
	f(2) = 1.78141141239d-9             !(TUNL)
	if (nchrat.ne.0) then
	  drate(2,1)=-5.d-3
	  drate(2,2)=5.d-3
	endif
C.......Li8 -> e- + v + 2He4.......(Ajzenberg-Selove 1988)
	f(3)  = 8.27d-1
C.......B12 -> e- + v + C12........(Ajzenberg-Selove 1990)
	f(4)  = 3.43d1
C.......C14 -> e- + v + N14........(Ajzenberg-Selove 1986)
	f(5)  = 3.834d-12
C.......B8 -> e+ + v + 2He4........(Ajzenberg-Selove 1988)
	f(6)  = 9.d-1
C.......C11 -> e+ + v + B11........(Ajzenberg-Selove 1990)
	f(7)  = 5.668d-4
C.......N12 -> e+ + v + C12........(Ajzenberg-Selove 1990)
	f(8)  = 6.301d1
C.......N13 -> e+ + v + C13........(Ajzenberg-Selove 1986)
	f(9)  = 1.159d-3
C.......O14 -> e+ + v + N14........(Ajzenberg-Selove 1986)
	f(10) = 9.8171d-3
C.......O15 -> e+ + v + N15........(Ajzenberg-Selove 1986)
	f(11) = 5.6704d-3

C-----Small network

C-----Temperature factors
      t913  = t9**(.33333333d0)           !t9**(1/3)
      t923  = t913*t913                   !t9**(2/3)
      t943  = t923*t923                   !t9**(4/3)
      t953  = t9*t923                     !t9**(5/3)
      t973  = t953*t923                   !t9**(7/3)
      t983  = t953*t9                     !t9**(8/3)
      t9103 = t973*t9                     !t9**(10/3)
      t9113 = t9103*t913                  !t9**(11/3)
      t9133 = t9113*t923                  !t9**(13/3)
      t9143 = t9133*t913                  !t9**(14/3)
      t9163 = t9133*t9                    !t9**(16/3)
      t9173 = t9163*t913                  !t9**(17/3)
      t9193 = t9173*t923                  !t9**(19/3)
      t912  = dsqrt(t9)                   !t9**(1/2)
      t932  = t9*t912                     !t9**(3/2)
      t952  = t932*t9                     !t9**(5/2)
      t972  = t952*t9                     !t9**(7/2)
      t992  = t972*t9                     !t9**(9/2)
      t9112 = t992*t9                     !t9**(11/2)
      t9132 = t9112*t9                    !t9**(13/2)
      t9152 = t9132*t9                    !t9**(15/2)
      t9m1  = 1/t9                        !t9**(-1)
      t9m13 = 1.d0/t913                   !t9**(-1/3)
      t9m23 = 1.d0/t923                   !t9**(-2/3)
      t9m12 = 1.d0/t912                   !t9**(-1/2)
      t9m32 = 1.d0/t932                   !t9**(-3/2)
      t9a   = t9/(1.d0+13.076d0*t9)       !For reaction 17.
      t9a32 = t9a**(3.d0/2.)              !t9a**(3/2)
      t9b   = t9/(1.d0+49.18d0*t9)        !For reaction 18.
      t9b32 = t9b**(1.5d0)                !t9b**(3/2)
      if (t9.gt.10.d0) then               !For reaction 22.
        t9c = 1.d0
      else
        t9c=t9/(1.-9.69d-2*t9+2.84d-2*t953/(1.d0-9.69d-2*t9)**(2.d0/3.))
      endif

      t9c13 = t9c**(1./3.d0)              !t9c**(1/3)
      t9c56 = t9c**(.8333333d0)           !t9c**(5/6)
      t9d   = t9/(1.d0+0.759d0*t9)        !For reaction 24.
      t9d13 = t9d**(1./3.d0)              !t9d**(1/3)
      t9d56 = t9d**(.8333333d0)           !t9d**(5/6)
      t9e   = t9/(1.d0+0.1378d0*t9)       !For reaction 26.
      t9e13 = t9e**(1./3.d0)              !t9e**(1/3)
      t9e56 = t9e**(.8333333d0)           !t9e**(5/6)
      t9f   = t9/(1.d0+0.1071d0*t9)       !For reaction 27.
      t9f13 = t9f**(1./3.d0)              !t9f**(1/3)
      t9f56 = t9f**(.8333333d0)           !t9f**(5/6)

C-----Neutron, photon reactions

C.......H(n,g)H2...................(Serpico et al 2004)
	if (t9.le.1.5d0) then
	  f(12)=44060.*(1.d0 - 2.7503695564153143*t9 - 
     -    3.5220409333117897*t9**2 - 0.2093513619089196*t9**3 + 
     -    0.10659679579058313*t912 + 4.62948586627009*t932 + 
     -    1.3459574632779876*t952)
	  df=67.95212182165885*dsqrt(1.255364135304537d0 + 
     -    618.8966688079934*t9 + 21793.605102078516*t9**2 + 
     -    97573.42092925594*t9**3 + 85539.62453206794*t9**4 + 
     -    14687.125326371255*t9**5 + 247.57751149254238*t9**6 - 
     -    2855.9302137640752*t9112 - 25.82889427679147*t912 - 
     -    5043.279674275595*t932 - 57307.155278753264*t952 - 
     -    111071.90152148822*t972 - 44154.571125115704*t992)
	else
	  f(12)=4.742d4*(1.d0-.8504d0*t912+.4895d0*t9-.09623d0*t932+
     -    8.471d-3*t9**2-2.80d-4*t952)	!SKM
	endif
	if (nchrat.ne.0) then
	  if (t9.le.1.5d0) then
	    drate(12,1)=-df/f(12)
	  else
	    drate(12,1)=-.078d0
	  endif
	  drate(12,2)=-drate(12,1)
	endif

C.......H2(n,g)H3..................(Wagoner 1969)
	f(13)  = 6.62d1*(1.d0+18.9d0*t9)
	if (nchrat.ne.0) then
	  drate(13,1)=-.3d0                 !1xFCZI
	  drate(13,2)=.3d0
	endif

C.......He3(n,g)He4................(Wagoner 1969)
	f(14)  = 6.62d0*(1.d0+905.d0*t9)
	if (nchrat.ne.0) then
	  drate(14,1)=-.5d0                 !1xFCZI
	  drate(14,2)=1.d0
	endif

C.......Li6(n,g)Li7................(Malaney-Fowler 1989)
	f(15)  = 5.10d3
	if (nchrat.ne.0) then
	  drate(15,1)=-.3d0                 !1xFCZI
	  drate(15,2)=.3d0
	endif

C-----Neutron, proton reactions

C.......He3(n,p)H3.................(Serpico et al 2004)
	if (t9.le.2.5d0) then
	  f(16)=7.064935d8 + 6.733213571736319d8*t9 + 
     -  1.7181155480346258d9*t9**2 - 4.5367658146835446d8*t9**3 - 
     -  1.2216728981712557d8*t9**4 - 4.92736677238425d8*t912 - 
     -  1.3659670893994067d9*t932 - 6.629932739639357d8*t952 + 
     -  4.834951929033479d8*t972
	  df=dsqrt(3.51d11 + 3.112097416989299d11*t9 + 
     -   7.901575411070865d10*t9**2 + 2.04541951561715d10*t9**3 + 
     -   5.110885965380451d9*t9**4 + 3.9016700171412725d9*t9**5 + 
     -   1.2106464640648174d9*t9**6 + 2.842691804858251d8*t9**7 + 
     -   2.5025023636054292d8*t9**8 - 1.0919522573895195d9*t9112 - 
     -   5.074476577064073d11*t912 - 9.073561744271307d8*t9132 - 
     -   4.935780126698165d8*t9152 - 1.3272119856586942d11*t932 - 
     -   3.982502921484235d10*t952 - 1.4832025658250046d10*t972 - 
     -   3.9093487936349277d9*t992)
	else
	  f(16)=4.81732d8
	  df=224626.d0
	endif
	if (nchrat.ne.0) then
	  drate(16,1)=-dsqrt(.00187d0**2+5.4d0*(df/f(16))**2)
	  drate(16,2)=-drate(16,1)
	endif

C.......Be7(n,p)Li7................(Serpico et al 2004)
	if (t9.le.2.5d0) then
	  f(17)=6.8423032d9 + 1.7674863d10*t9 + 2.6622006d9*t9**2 - 
     -  3.3561608d8*t9**3 - 5.9309139d6*t9**4 - 1.4987996d10*t912 - 
     -  1.0576906d10*t932 + 2.7447598d8*t952 + 7.6425157d7*t972 - 
     -  (2.282944d7*t9m32)/ex(0.050351813d0/t9)
	  fp=6.8637241d9 + 1.7654706d10*t9 + 2.6528695d9*t9**2 - 
     -  3.334255d8*t9**3 - 5.8841835d6*t9**4 - 
     -  1.4998087d10*t912 - 1.0549873d10*t932 + 
     -  2.7244147d8*t952 + 7.5882824d7*t972 - 
     -  (2.2946239d7*t9m32)/ex(0.05042127d0/t9)
	  fm=5.3356377d9 + 1.2258644d10*t9 + 1.6991251d9*t9**2 - 
     -  2.011188d8*t9**3 - 3.0052814d6*t9**4 - 1.0649687d10*t912 - 
     -  7.1221438d9*t932 + 1.8848519d8*t952 + 4.2410535d7*t972 + 
     -  (6.0102571d7*t9m32)/ex(0.2761375d0/t9)
	else
	  f(17)=1.28039d9
	  fp=1.28652d9
	  fm=1.27454d9
	endif
	if (nchrat.ne.0) then
	  drate(17,1)=-dsqrt(.02082d0**2+1.2d0*(fm/f(17)-1.d0)**2)
	  drate(17,2)=dsqrt(.02082d0**2+1.2d0*(fp/f(17)-1.d0)**2)
	endif

C-----Neutron, alpha reactions

C.......Li6(n,t)He4................(Caughlan-Fowler 1988)
	f(18)  = 2.54d9*t9m32*ex(-2.39d0/t9)+
     .    1.68d8*(1.d0-.261d0*t9b32/t932)
	if (nchrat.ne.0) then
	  drate(18,1)=-.1d0                 !1xFCZI
	  drate(18,2)=.1d0
	endif

C.......Be7(n,a)He4................(Wagoner 1969)
	f(19)  = 2.05d4*(1.d0+3760.d0*t9)
	if (nchrat.ne.0) then
	  drate(19,1)=-.9d0
	  drate(19,2)=9.d0
	endif

C-----Proton, proton reactions
	if(dpg.eq.'PIS2020') then

C.......H2(p,g)He3.................(Pisanti et al 2020)
	if (t9.le.4.d0) then
	  f(20)=
     -    ((-6.194453277203594d0+22171.926490994258d0*t9-
     -      812513.0306179419d0*t9**2+1.7722520963024362d6*t9**3-
     -      368351.0375876271d0*t9**4+5275.48295368989d0*t9**5-
     -      1.3641334338787347d6*t9103+810606.1958071451d0*t9113+
     -      201.7432760051548d0*t913+125602.00856877993d0*t9133-
     -      31095.323648099842d0*t9143-548.4032679748573d0*t9163+
     -      26.334040198363052d0*t9173-2817.4051296076195d0*t923-
     -      109072.97636789024d0*t943+355668.0024142403d0*t953+
     -      1.3796464841892694d6*t973-1.7815928451228556d6*t983)*
     -      t9m23)/ex(1.29042942*t9m13)
	  fp=((-2.6703059325264484d0+13244.37219953217d0*t9-
     -      434602.9565597139d0*t9**2+633398.9523744078d0*t9**3-
     -      60428.796285877674d0*t9**4+110.22914339351317d0*t9**5-
     -      398185.74145106773d0*t9103+183727.33326277588d0*t9113+
     -      103.19823960924714d0*t913+13409.7312116957d0*t9133-
     -      1799.1783726456658d0*t9143-1597.1426434352331d0*t923-
     -      65649.67419229684d0*t943+206135.95747657356d0*t953+
     -      663735.6949855759d0*t973-750238.7765562678d0*t983)*
     -      t9m23)/ex(1.29042942*t9m13)
	  fm=((-2.5097249605312286d0+12447.911377938311d0*t9-
     -      408467.7632418989d0*t9**2+595309.003680159d0*t9**3-
     -      56794.86257556095d0*t9**4+103.60042597400357d0*t9**5-
     -      374240.5257062942d0*t9103+172678.7440993883d0*t9113+
     -      96.99233152266518d0*t913+12603.326355107645d0*t9133-
     -      1690.983348297709d0*t9143-1501.097203707621d0*t923-
     -      61701.779004447315d0*t943+193739.80829906635d0*t953+
     -      623821.4226376609d0*t973-705122.5725742867d0*t983)*
     -      t9m23)/ex(1.29042942*t9m13)
	else
	  f(20)=2113.16d0
	  fp=2178.67d0
	  fm=2047.65d0
	endif
	if (nchrat.ne.0) then
	  drate(20,1)=fm/f(20)-1.d0
	  drate(20,2)=fp/f(20)-1.d0
	endif

	elseif(dpg.eq.'LUNA2019') then

C.......H2(p,g)He3.................(LUNA, Nature 2020)
	if (t9.le.4.d0) then
	  f(20)=
     -    ((7.090200718076673d0-3711.154677141494d0*t9+
     -    2629.23222805383d0*t9**2-101.39648022497795d0*t9**3-
     -    125.97050644788978d0*t913+972.6533904929203d0*t923+
     -    6567.1112067593d0*t943-4407.983553142122d0*t953-
     -    826.1451681441783d0*t973+315.48118933726647d0*t983)*
     -    t9m23)/ex(1.29042942d0*t9m13)
	  fp=
     -    ((16.3032734879185d0-5842.673850513178d0*t9+
     -    5519.545942677666d0*t9**2-103.79573790323661d0*t9**3-
     -    250.2164363577417d0*t913+1670.7185564253375d0*t923+
     -    10428.154033816412d0*t943-8674.399414433765d0*t953-
     -    1920.2531479346424d0*t973+515.9295068803921d0*t983)*
     -    t9m23)/ex(1.29042942d0*t9m13)
	  fm=
     -    ((-2.1001919572699803d0-1581.8646794305341d0*t9-
     -    259.4143952854345d0*t9**2-99.00742867019478d0*t9**3-
     -    1.9515175015352444d0*t913+275.5441321744908d0*t923+
     -    2709.247601272416d0*t943-144.45079847595105d0*t953+
     -    267.3684117834248d0*t973+115.15180816889564d0*t983)*
     -    t9m23)/ex(1.29042942d0*t9m13)
	else
	  f(20)=2111.85d0
	  fp=2277.78d0
	  fm=1945.91d0
	endif
	if (nchrat.ne.0) then
	  drate(20,1)=-dsqrt(.022d0**2+(fm/f(20)-1.d0)**2)
	  drate(20,2)=dsqrt(.022d0**2+(fp/f(20)-1.d0)**2)
	endif

	elseif(dpg.eq.'IL2016') then

C.......H2(p,g)He3.................(Iliadis et al 2016)
C....The fit has a precision of better than 0.2%
	if (t9.le.4.d0) then
	  f(20)=
     -    ((55.55153833295411d0-45332.4349427496d0*t9+
     -      360349.06069080054d0*t9**2-88274.4582154477d0*t9**3+
     -      21230.83366308875d0*t9103-2279.6689031328287d0*t9113-
     -      1120.408092977454d0*t913+9549.43975384368d0*t923+
     -      133855.95688348415d0*t943-263167.9922114686d0*t953-
     -      338400.75767304475d0*t973+214892.01141493738d0*t983)*
     -    t9m23)/ex(1.29042942d0*t9m13)
	  fp=1.037d0*f(20)
	  fm=.963d0*f(20)
	else
	  f(20)=2241.0d0
	  fp=1.037d0*f(20)
	  fm=.963d0*f(20)
	endif
	if (nchrat.ne.0) then
	  drate(20,1)=fm/f(20)-1.d0
	  drate(20,2)=fp/f(20)-1.d0
	endif

	elseif(dpg.eq.'MARCII') then

C.......H2(p,g)He3.................(Marcucci et al 2016)
C....The uncertainties have been taken like in Coc2015
	if (t9.le.4.d0) then
	  f(20)=((-379.7864935700319d0+87196.10500012108d0*t9-
     -      176962.2342450312d0*t9**2+5480.572151978118d0*t9**3-
     -      439.2151087880425d0*t9103+4988.771589356891d0*t913-
     -      27853.19243579095d0*t923-170483.22744874505d0*t943+
     -      217140.9218402939d0*t953+92790.36482743552d0*t973-
     -      30024.90894289211d0*t983)*
     -    t9m23)/ex(1.29042942d0*t9m13)
	  fp=1.05d0*f(20)
	  fm=.95d0*f(20)
	else
	  f(20)=2327.04d0
	  fp=1.05d0*f(20)
	  fm=.95d0*f(20)
	endif
	if (nchrat.ne.0) then
	  drate(20,1)=fm/f(20)-1.d0
	  drate(20,2)=fp/f(20)-1.d0
	endif

	elseif(dpg.eq.'COC2015') then

C.......H2(p,g)He3.................(Coc et al 2015)
	if (t9.le.4.d0) then
	  f(20)=-9187.46958169868d0*t9+1.0444856137407974d6*t9**2-
     -  3.9945879671129556d6*t9**3+936107.4601901153d0*t9**4+
     -  63721.22895814322d0*t9**4.666666666666667d0-
     -  8175.550931523481d0*t9**5+
     -  476.26978096798865d0*t9**5.333333333333333d0+
     -  3.3542961405502073d6*t9103-2.077144726435762d6*t9113-
     -  20.556313115404247d0*t913-298387.1261724728d0*t9133+
     -  670.6194906656923d0*t923+70087.49782324153d0*t943-
     -  332770.31917676615d0*t953-2.263585691707297d6*t973+
     -  3.5143875112608057d6*t983
	  fp=1.05d0*f(20)
	  fm=.95d0*f(20)
	else
	  f(20)=2244.11d0
	  fp=1.05d0*f(20)
	  fm=.95d0*f(20)
	endif
	if (nchrat.ne.0) then
	  drate(20,1)=fm/f(20)-1.d0
	  drate(20,2)=fp/f(20)-1.d0
	endif

	elseif(dpg.eq.'MARCI') then

C.......H2(p,g)He3.................(Marcucci et al 2005)
C....The uncertainties have been taken like in Coc2015
	if (t9.le.4.d0) then
	  f(20)=((62.50292299762131d0-9046.970544787797d0*t9+
     -      7268.776629314168d0*t9**2-46.563648896376414d0*t9**3-
     -      684.7528248901089d0*t913+3313.003452986549d0*t923+
     -      13993.157194545502d0*t943-11361.91926687308d0*t953-
     -      2681.334449686284d0*t973+552.9301135160282d0*t983)*
     -    t9m23)/ex(1.29042942d0*t9m13)
	  fp=1.05d0*f(20)
	  fm=.95d0*f(20)
	else
	  f(20)=2284.37d0
	  fp=1.05d0*f(20)
	  fm=.95d0*f(20)
	endif
	if (nchrat.ne.0) then
	  drate(20,1)=fm/f(20)-1.d0
	  drate(20,2)=fp/f(20)-1.d0
	endif

	elseif(dpg.eq.'AD2011') then

C.......H2(p,g)He3.................(Adelberger et al 2011)
	if (t9.le.4.d0) then
	  f(20)=((-7.613432267854739d0+
     -    42.56993228087691d0*t9**0.3333333333333333d0+
     -    158.34997086013004d0*t9**0.6666666666666666d0-
     -    1570.0694191913508d0*t9+
     -    3240.027014170479d0*t9**1.3333333333333333d0-
     -    1154.0660803376381d0*t9**1.6666666666666667d0+
     -    383.3696816350286d0*t9**2+
     -    185.50354727308456d0*t9**2.3333333333333335d0-
     -    35.125998466685d0*t9**2.6666666666666665d0+
     -    3.286192216969564d0*t9**3)*
     -    t9m23)/ex(1.29042942d0*t9m13)
	  fp=((-12.49923409629059d0+
     -    96.35202573638982d0*t9**0.3333333333333333d0 - 
     -    70.36614651647116d0*t9**0.6666666666666666d0 - 
     -    1115.8232439446347d0*t9 + 
     -    2832.8535084444093d0*t9**1.3333333333333333d0 - 
     -    1004.1133868889383d0*t9**1.6666666666666667d0 + 
     -    416.6581623537144d0*t9**2 + 
     -    262.89416081650864d0*t9**2.3333333333333335d0 - 
     -    44.90571861171566d0*t9**2.6666666666666665d0 + 
     -    3.991309604081436d0*t9**3)*
     -      t9m23)/ex(1.29042942d0*t9m13)
	  fm=((-3.4460815944902d0-
     -    2.923755296618394d0*t9**0.3333333333333333d0+
     -    348.7231833306605d0*t9**0.6666666666666666d0-
     -    1936.5501516662362d0*t9+
     -    3550.185814925816d0*t9**1.3333333333333333d0-
     -    1263.6638139110366d0*t9**1.6666666666666667d0+
     -    346.9151426024522d0*t9**2+
     -    118.92339010258894d0*t9**2.3333333333333335d0-
     -    26.686544171196807d0*t9**2.6666666666666665d0+
     -    2.6727583442238463d0*t9**3)*
     -    t9m23)/ex(1.29042942d0*t9m13)
	else
	  f(20)=2245.91d0
	  fp=2667.71d0
	  fm=1865.19d0
	endif
	if (nchrat.ne.0) then
	  drate(20,1)=fm/f(20)-1.d0
	  drate(20,2)=fp/f(20)-1.d0
	endif

	else
	  write(*,*) 'error in dpg choice'
	  stop
	endif

C.......H3(p,g)He4.................(Caughlan-Fowler 1988)
	f(21)  = 2.2d4*t9m23*ex(-3.869d0/t913)*
     .    (1.d0+.108d0*t913+1.68d0*t923+1.26d0*t9+
     .    .551d0*t943+1.06d0*t953)
	if (nchrat.ne.0) then
	  drate(21,1)=-.2d0                 !1xFCZI
	  drate(21,2)=.2d0
	endif

C.......Li6(p,g)Be7................(NACRE 1999)
	f(22) = 1.25d6*t9m23*ex(-8.415d0/t913)*
     .    (1.d0-.252d0*t9+5.19d-2*t9**2-2.92d-3*t9**3)	 
	if (nchrat.ne.0) then
	  drate(22,1)=-.21d0
	  drate(22,2)=.21d0
	endif

C-----Proton, alpha reactions

C.......Li6(p,He3)He4..............(Serpico et al 2004)
	if (t9.le.2.5d0) then
	  f(23)=((-7.4966212d7 - 1.9411561d10*t9 + 
     -      1.6262854d10*t9**2 + 2.0533495d7*t913 + 
     -      3.9547491d9*t923 + 3.7907358d10*t943 - 
     -      3.4313768d10*t953 - 3.9965228d9*t973 + 
     -      4.0333873d8*t983)*t9m23)/ex(4.62619323d0*t9m13)
	  fp=((4.6794127d7 - 1.5077363d9*t9 - 2.8610381d9*t9**2 - 
     -      4.1883216d8*t913 + 1.3422134d9*t923 - 9.4597359d8*t943+
     -      3.6073249d9*t953 + 9.4073567d8*t973 - 1.1547155d8*t983)*
     -      t9m23)/ex(2.47110932d0*t9m13)
	  fm=((-2.1541443d7 - 2.1453941d10*t9 + 1.5165239d10*t9**2-
     -      4.5939493d8*t913 + 5.5208615d9*t923 + 
     -      3.8266784d10*t943 - 3.3068204d10*t953 - 
     -      3.6300979d9*t973 + 3.5841419d8*t983)*t9m23)/
     -      ex(4.53459377d0*t9m13)
	else
	  f(23)=3.05102d7
	  fp=3.09863d7
	  fm=3.00895d7
	endif
	if (nchrat.ne.0) then
	  drate(23,1)=-dsqrt(.0931d0**2+1.2d0*(fm/f(23)-1.d0)**2)
	  drate(23,2)=dsqrt(.0931d0**2+1.2d0*(fp/f(23)-1.d0)**2)
	endif

C.......Li7(p,a)He4................(Serpico et al 2004)
	if (t9.le.2.5d0) then
	  f(24)=((-8.9654123d7 - 2.5851582d8*t9 - 2.6831252d7*t9**2 + 
     -      3.8691673d8*t913 + 4.9721269d8*t923 + 
     -      2.6444808d7*t943 - 1.2946419d6*t953 - 
     -      1.0941088d8*t973 + 9.9899564d7*t983)*t9m23)/
     -      ex(7.73389632d0*t9m13)
	  fp=((1.6425644d7 - 7.682657d8*t9 + 1.2461811d9*t9**2 - 
     -      1.1914365d8*t913 + 3.3659333d8*t923 + 1.8234158d9*t943 - 
     -      1.9962683d9*t953 - 5.4978741d8*t973 + 1.4214466d8*t983)*
     -      t9m23)/ex(6.34172901d0*t9m13)
	  fm=((-2.9979375d6 - 7.8110137d8*t9 + 
     -      1.1816185d9*t9**2 + 5.700657d6*t913 + 
     -      1.330785d8*t923 + 1.733923d9*t943 - 
     -      1.8284296d9*t953 - 5.0363158d8*t973 + 
     -      1.1026194d8*t983)*t9m23)/ex(5.35732631d0*t9m13)
C-----Be8 resonance contribution to Li7(p,a)He4
	  f(24)=f(24)+ex(-1.137519d0*t9**2 - 8.6256687d0*t9m13)*
     -  (3.0014189d7 - 1.8366119d8*t9 + 1.7688138d9*t9**2 - 
     -    8.4772261d9*t9**3 + 2.0237351d10*t9**4 - 
     -    1.9650068d10*t9**5 + 7.9452762d8*t9**6 + 
     -    1.3132468d10*t9**7 - 8.209351d9*t9**8 - 9.1099236d8*t9**9 + 
     -    2.7814079d9*t9**10 - 1.0785293d9*t9**11 + 
     -    1.3993392d8*t9**12)*t9m23
	  fp=fp+ex(-1.0418442d0*t9**2 - 5.5570697d0*t9m13)*
     -  (-25145.507d0 + 1.0787318d6*t9 - 1.5899728d7*t9**2 + 
     -    1.7182625d8*t9**3 - 8.3103078d8*t9**4 + 2.1243451d9*t9**5 - 
     -    2.872313d9*t9**6 + 2.0104043d9*t9**7 - 4.3859588d8*t9**8 - 
     -    3.529339d8*t9**9 + 2.9815567d8*t9**10 - 
     -    8.8920729d7*t9**11 + 9.9850915d6*t9**12)*t9m23
	  fm=fm+ex(-1.0068557d0*t9**2 - 5.2092464d0*t9m13)*
     -  (-14997.544d0 + 665017.06d0*t9 - 1.0880148d7*t9**2 + 
     -    1.1299875d8*t9**3 - 5.3097151d8*t9**4 + 1.3288827d9*t9**5 - 
     -    1.7652952d9*t9**6 + 1.2196578d9*t9**7 - 2.6871614d8*t9**8 - 
     -    2.0119802d8*t9**9 + 1.7032325d8*t9**10 - 
     -    5.0416533d7*t9**11 + 5.6188182d6*t9**12)*t9m23
	else
	  f(24)=1.53403d6
	  fp=1.57087d6
	  fm=1.49673d6
C-----Be8 resonance contribution to Li7(p,a)He4
	  f(24)=f(24)+84516.7d0
	  fp=fp+85552.6d0
	  fm=fm+83165.1d0
	endif
	if ((nchrat.ne.0).and.(t9.le.10.d0)) then
	  drate(24,1)=-dsqrt(.08d0**2+(fm/f(24)-1.d0)**2)
	  drate(24,2)=dsqrt(.08d0**2+(fp/f(24)-1.d0)**2)
	endif

C-----Alpha, proton reactions

C.......He4(d,g)Li6................(NACRE 1999)
	f(25) = 1.482d1*t9m23*ex(-7.435d0/t913)*
     .    (1.d0+6.572d0*t9+7.6d-2*t9**2+2.48d-2*t9**3)+
     .    8.28d1*t9m32*ex(-7.904d0/t9)
	if ((nchrat.ne.0).and.(t9.le.10.d0)) then
	  drate(25,1)=-.04d0-.9413d0+.355d0*t912-.0411d0*t9
	  drate(25,2)=.04d0+.249d0+5.612d0*ex(-3.d0*t9)-
     .      2.63d0*ex(-2.d0*t9)+.773d0*ex(-t9)
	endif

C.......He4(t,g)Li7................(Serpico et al 2004)
	if (t9.le.2.5d0) then
	  f(26)=((0.094614248d0 - 4.9273133d0*t9 + 99.358965d0*t9**2 - 
     -      989.81236d0*t9**3 + 4368.45d0*t9**4 + 
     -      931.93597d0*t9**5 - 391.07855d0*t9**6 + 159.23101d0*t9**7 - 
     -      34.407594d0*t9**8 + 3.3919004d0*t9**9 + 
     -      0.017556217d0*t9**10 - 0.036253427d0*t9**11 + 
     -      0.0031118827d0*t9**12 - 0.00008714468d0*t9**13)*t9m12)/
     -      (ex(8.4d-7*t9)*(1.d0 + 1.78616593d0*t9)**3)
	  fp=((0.083877015d0 - 4.5408918d0*t9 + 96.316095d0*t9**2 - 
     -      1016.5548d0*t9**3 + 4809.4834*t9**4 - 
     -      168.10236*t9**5 + 208.81839d0*t9**6 - 64.618239d0*t9**7 + 
     -      10.478926d0*t9**8 - 0.41782376d0*t9**9 - 
     -      0.06453532d0*t9**10 + 0.004777625d0*t9**11 + 
     -      0.00020027244d0*t9**12 - 0.000017864206d0*t9**13)*t9m12)/
     -      (ex(9.3d-7*t9)*(1.d0 + 1.60170507d0*t9)**3)
	  fm=((0.066096606d0 - 3.5622862d0*t9 + 75.13824d0*t9**2 - 
     -      788.24146d0*t9**3 + 3705.8889d0*t9**4 - 106.98552d0*t9**5 + 
     -      139.5561d0*t9**6 - 7.8984539d0*t9**7 - 
     -      1.6035703d0*t9**8 - 0.17508886d0*t9**9 + 
     -      0.046425912d0*t9**10 + 0.0030233156d0*t9**11 - 
     -      0.00081682606d0*t9**12 + 0.000034545163d0*t9**13)*
     -      t9m12)/(ex(0.01111331d0*t9)*(1.d0 + 1.63277688d0*t9)**3)
	else
	  f(26)=807.406d0
	  fp=916.513d0
	  fm=698.635d0
	endif
	if (nchrat.ne.0) then
	  drate(26,1)=-dsqrt(.0871d0**2+(fm/f(26)-1.d0)**2)
	  drate(26,2)=dsqrt(.0871d0**2+(fp/f(26)-1.d0)**2)
	endif
	
C.......He4(He3,g)Be7..............(Serpico et al 2004)
	if (t9.le.2.5d0) then
	  f(27)=((0.000046165644d0 - 0.00046036111d0*t9 - 
     -      0.021600946d0*t9**2 + 0.069627779d0*t9**3 + 
     -      7.346612d0*t9**4 - 95.123199d0*t9**5 + 
     -      391.13123d0*t9**6 - 187.23717d0*t9**7 + 
     -      86.111544d0*t9**8 - 21.630169d0*t9**9 + 3.6006922d0*t9**10 - 
     -      0.34322836d0*t9**11 + 0.018106742d0*t9**12 - 
     -      0.00035681506d0*t9**13)*t9m12)/
     -      (ex(0.48102949d0*t9)*(1.d0 + 1.17917554d0*t9)**3)
	  fp=((0.000050712746d0 - 0.00048202784d0*t9 - 0.023831596d0*t9**2 + 
     -      0.056033679d0*t9**3 + 8.408972d0*t9**4 - 106.22688d0*t9**5 + 
     -      434.78964d0*t9**6 - 238.48007d0*t9**7 + 
     -      94.757251d0*t9**8 - 23.705813d0*t9**9 + 3.8007127d0*t9**10 - 
     -      0.37029512d0*t9**11 + 0.019933598d0*t9**12 - 
     -      0.00045281691d0*t9**13)*t9m12)/
     -      (ex(0.2282385d0*t9)*(1.d0 + 1.31654256d0*t9)**3)
	  fm=((0.000049798665d0 - 0.00047801278d0*t9 - 
     -      0.023362423d0*t9**2 + 0.061790614d0*t9**3 + 
     -      8.0589558d0*t9**4 - 102.19595d0*t9**5 + 418.23687d0*t9**6 - 
     -      229.34858d0*t9**7 + 92.638713d0*t9**8 - 23.370595d0*t9**9 + 
     -      3.7644261d0*t9**10 - 0.36726621d0*t9**11 + 
     -      0.019784483d0*t9**12 - 0.00044951929d0*t9**13)*t9m12)/
     -      (ex(0.25325444d0*t9)*(1.d0 + 1.28569931d0*t9)**3)
	else
	  f(27)=149.06d0
	  fp=150.371d0
	  fm=147.698d0
	endif
	if (nchrat.ne.0) then
	  drate(27,1)=-dsqrt(.048d0**2+2.1d0*(fm/f(27)-1.d0)**2)
	  drate(27,2)=dsqrt(.048d0**2+2.1d0*(fp/f(27)-1.d0)**2)
	endif

C-----Deuterium, neutron and deuterium, proton reactions
	if(ddn.eq.'PIS2020') then

C.......H2(d,n)He3.................(Pisanti et al 2020)
	if (t9.le.4.d0) then
	  f(28)=(321296.9948970247d0-1.9180215905621612d9*t9+
     -      1.3881096257595047d11*t9**2-
     -      5.289640171653128d11*t9**3+1.719654390000693d11*t9**4-
     -      3.699206623147024d9*t9**5+4.760387471479396d11*t9103-
     -      3.2811985153873035d11*t9113-1.2209299424235549d7*t913-
     -      6.734692924560129d10*t9133+
     -      1.9089695890090744d10*t9143+4.384049981959462d8*t9163-
     -      2.396477962456513d7*t9173+2.018310465362159d8*t923+
     -      1.1672167355300453d10*t943-4.807140317985709d10*t953-
     -      2.8968459770013477d11*t973+4.496605591595117d11*t983)*
     -      t9m23/ex(t9m13)
	  fp=(324509.96470113087d0-1.937201805966554d9*t9+
     -      1.40199072171838d11*t9**2-5.3425365720533984d11*t9**3+
     -      1.7368509333437854d11*t9**4-3.736198687778469d9*t9**5+
     -      4.807991344908736d11*t9103-
     -      3.3140104995728906d11*t9113-1.2331392412242804d7*t913-
     -      6.802039851406439d10*t9133+
     -      1.9280592841500507d10*t9143+
     -      4.4278904796883184d8*t9163-2.420442740820764d7*t9173+
     -      2.0384935694489682d8*t923+1.1788889026117342d10*t943-
     -      4.855211720111187d10*t953-2.925814436133358d11*t973+
     -      4.54157164646803d11*t983)*t9m23/ex(t9m13)
	  fm=(318084.0248363851d0-1.8988413743645012d9*t9+
     -      1.3742285293887856d11*t9**2-
     -      5.2367437697077356d11*t9**3+
     -      1.7024578461115326d11*t9**4-3.662214557228816d9*t9**5+
     -      4.712783596626029d11*t9103-3.248386530191517d11*t9113-
     -      1.2087206428878365d7*t913-6.6673459955072395d10*t9133+
     -      1.8898798932232002d10*t9143+
     -      4.3402094826630807d8*t9163-2.3725131832133267d7*t9173+
     -      1.998127360459775d8*t923+1.1555445680334196d10*t943-
     -      4.759068914333845d10*t953-2.867877517033764d11*t973+
     -      4.451639535427516d11*t983)*t9m23/ex(t9m13)
	else
	  f(28)=5.02027d7
	  fp=5.07047d7
	  fm=4.97006d7
	endif
	if ((nchrat.ne.0).and.(t9.le.10.d0)) then
	  drate(28,1)=fm/f(28)-1.d0
	  drate(28,2)=fp/f(28)-1.d0
	endif

	elseif(ddn.eq.'PIS2020noTH') then

C.......H2(d,n)He3.................(Pisanti et al 2020, no TH)
	if (t9.le.4.d0) then
	  f(28)=(317993.67295926367d0-1.8919997128681066d9*t9+
     -      1.3640240871437564d11*t9**2-
     -      5.182710058060936d11*t9**3+
     -      1.6825750321490686d11*t9**4-
     -      3.6165673562135224d9*t9**5+4.661526394812192d11*t9103-
     -      3.211619903561985d11*t9113-1.2070808939234572d7*t913-
     -      6.5874673097090675d10*t9133+
     -      1.8667495585769524d10*t9143+
     -      4.2852468770347506d8*t9163-2.3420462523899723d7*t9173+
     -      1.9932206266351324d8*t923+1.1499746982779984d10*t943-
     -      4.729989325203574d10*t953-2.843112343767856d11*t973+
     -      4.408935520503923d11*t983)*t9m23/ex(t9m13)
	  fp=(321173.60962526774d0-1.9109197097254534d9*t9+
     -      1.3776643278217865d11*t9**2-
     -      5.2345371576422455d11*t9**3+
     -      1.699400781986653d11*t9**4-3.652733028219671d9*t9**5+
     -      4.708141657736811d11*t9103-3.24373610179118d11*t9113-
     -      1.2191517018158576d7*t913-6.653341980636526d10*t9133+
     -      1.8854170534594234d10*t9143+
     -      4.3280993437036836d8*t9163-2.365466713607255d7*t9173+
     -      2.0131528325620362d8*t923+1.1614744451035288d10*t943-
     -      4.777289217811496d10*t953-2.8715434667687427d11*t973+
     -      4.453024874955954d11*t983)*t9m23/ex(t9m13)
	  fm=(314813.7363106378d0-1.873079716069881d9*t9+
     -      1.3503838464970602d11*t9**2-
     -      5.1308829586023413d11*t9**3+
     -      1.6657492823561295d11*t9**4-
     -      3.5804016843146996d9*t9**5+4.614911132001915d11*t9103-
     -      3.17950370541469d11*t9113-1.1950100848217864d7*t913-
     -      6.521592638963368d10*t9133+1.848082063747956d10*t9143+
     -      4.242394410497244d8*t9163-2.3186257912466004d7*t9173+
     -      1.9732884206787813d8*t923+1.138474951482869d10*t943-
     -      4.682689432709231d10*t953-2.8146812208315826d11*t973+
     -      4.3648461661534406d11*t983)*t9m23/ex(t9m13)
	else
	  f(28)=5.11045d7
	  fp=5.16156d7
	  fm=5.05935d7
	endif
	if ((nchrat.ne.0).and.(t9.le.10.d0)) then
	  drate(28,1)=fm/f(28)-1.d0
	  drate(28,2)=fp/f(28)-1.d0
	endif

	elseif(ddn.eq.'GI2017') then

C.......H2(d,n)He3.................(Gomez Inesta et al 2017)
C....The fit has a precision of better than 0.1%
	if (t9.le.4.d0) then
	  f(28)=-7433.621994370127d0-1.6641604383377159d7*t9-
     -  2.199344170260687d9*t9**2-3.6529866713525444d10*t9**3-
     -  1.2045642271220781d11*t9**4-1.0691243627804767d11*t9**5-
     -  2.5114022323991688d10*t9**6-1.0932599052647448d9*t9**7+
     -  6.18664657581283d10*t9112+564495.3203623195d0*t912+
     -  6.782028700177776d9*t9132+7.952668177641357d7*t9152+
     -  2.5302576506644633d8*t932+1.1403138897300697d10*t952+
     -  7.88687539859757d10*t972+1.3308290032662541d11*t992
	  fp=1.011d0*f(28)
	  fm=.989d0*f(28)
	else
	  f(28)=5.031d7
	  fp=1.011d0*f(28)
	  fm=.989d0*f(28)
	endif
	if (nchrat.ne.0) then
	  drate(28,1)=fm/f(28)-1.d0
	  drate(28,2)=fp/f(28)-1.d0
	endif

	elseif(ddn.eq.'COC2015') then

C.......H2(d,n)He3.................(Coc et al 2015)
	if (t9.le.4.d0) then
	  f(28)=1.6786149847191703d7*t9-3.6991476328646164d10*t9**2+
     -  5.378309561094637d11*t9**3-4.217731845580232d11*t9**4-
     -  8.216440806870335d10*t9**4.666666666666667d0+
     -  2.226819654173513d10*t9**5-
     -  4.0884793296485105d9*t9**5.333333333333333d0+
     -  4.558401614434838d8*t9**5.666666666666667d0-
     -  2.331490779802186d7*t9**6-6.657232633590537d11*t9103+
     -  6.123999171249615d11*t9113-109461.3403770242d0*t913+
     -  2.1691310067974054d11*t9133+1.7504426730402112d6*t923-
     -  6.176413625102873d8*t943+6.488159443055595d9*t953+
     -  1.3206977015057251d11*t973-3.170481824243264d11*t983
	  fp=1.02d0*f(28)
	  fm=.98d0*f(28)
	else
	  f(28)=5.03127d7
	  fp=1.02d0*f(28)
	  fm=.98d0*f(28)
	endif
	if (nchrat.ne.0) then
	  drate(28,1)=fm/f(28)-1.d0
	  drate(28,2)=fp/f(28)-1.d0
	endif

	elseif(ddn.eq.'PIS2007') then

C.......H2(d,n)He3.................(Pisanti et al 2007)
	if (t9.le.4.d0) then
	  f(28)=((-1.8436156d6 - 6.1150115d7*t9 - 2.7251853d7*t9**2 - 
     -      2.2800422d6*t9**3 - 252433.58d0*t9**4 - 284357.41d0*t9103 + 
     -      906146.25d0*t9113 + 1.2270083d7*t913 - 1.3680884d7*t923 + 
     -      1.328894d8*t943 - 1.1916242d7*t953 + 
     -      8.3705218d6*t973 + 2.2357751d6*t983)*t9m23)/
     -      ex(1.*t9m13)
	  fp=((-1.7643388d6 - 6.0708618d7*t9 - 2.9398403d7*t9**2 - 
     -      1.6035942d6*t9**3 - 188921.3d0*t9**4 - 
     -      345931.36d0*t9103 + 684862.04d0*t9113 + 1.1654271d7*t913 - 
     -      1.2269338d7*t923 + 1.26615d8*t943 - 
     -      3.5412041d6*t953 + 6.1432859d6*t973 + 2.8526453d6*t983)*
     -      t9m23)/ex(1.*t9m13)
	  fm=((-1.7643388d6 - 6.0708618d7*t9 - 
     -      2.9398403d7*t9**2 - 1.6035942d6*t9**3 - 
     -      188921.3d0*t9**4 - 345931.36d0*t9103 + 
     -      684862.04d0*t9113 + 1.1654271d7*t913 - 
     -      1.2269338d7*t923 + 1.26615d8*t943 - 3.5412041d6*t953 + 
     -      6.1432859d6*t973 + 2.8526453d6*t983)*t9m23)/
     -      ex(1.*t9m13)
	else
	  f(28)=4.98099d7
	  fp=4.99298d7
	  fm=4.969d7
	endif
	if ((nchrat.ne.0).and.(t9.le.10.d0)) then
	  drate(28,1)=-dsqrt(.01d0**2+13.8d0*(fm/f(28)-1.d0)**2)
	  drate(28,2)=dsqrt(.01d0**2+13.8d0*(fp/f(28)-1.d0)**2)
	endif

	else
	  write(*,*) 'error in ddn choice'
	  stop
	endif

	if(ddp.eq.'PIS2020') then

C.......H2(d,p)H3..................(Pisanti et al 2020)
	if (t9.le.4.d0) then
	  f(29)=(344771.39066679927d0-2.1068827419353156d9*t9+
     -      1.565621182899997d11*t9**2-6.085012672776245d11*t9**3+
     -      1.9970133506639343d11*t9**4-4.320369220422077d9*t9**5+
     -      5.4972593314086273d11*t9103-
     -      3.8007648756248285d11*t9113-1.3201397617111376d7*t913-
     -      7.837778110396318d10*t9133+
     -      2.2258122388142662d10*t9143+
     -      5.1278962480559975d8*t9163-2.8069295382999137d7*t9173+
     -      2.1992915636979988d8*t923+1.2930592352633339d10*t943-
     -      5.3731389712860016d10*t953-3.294384822438947d11*t973+
     -      5.147142897092518d11*t983)*t9m23/ex(t9m13)
	  fp=(348219.1045496517d0-2.1279515692849133d9*t9+
     -      1.581277394695097d11*t9**2-6.145862799382203d11*t9**3+
     -      2.016983484128864d11*t9**4-4.363572912529403d9*t9**5+
     -      5.552231924611802d11*t9103-
     -      3.8387725243032263d11*t9113-1.3333411592163663d7*t913-
     -      7.91615589133291d10*t9133+2.2480703611537277d10*t9143+
     -      5.1791752104185057d8*t9163-2.8349988336165372d7*t9173+
     -      2.221284479263768d8*t923+1.305989827580598d10*t943-
     -      5.426870360871224d10*t953-3.3273286705956134d11*t973+
     -      5.198614325960014d11*t983)*t9m23/ex(t9m13)
	  fm=(341323.67680061463d0-2.0858139146439614d9*t9+
     -      1.5499649711361087d11*t9**2-
     -      6.024162546289459d11*t9**3+
     -      1.9770432172408563d11*t9**4-
     -      4.2771655284107037d9*t9**5+
     -      5.4422867383153156d11*t9103-
     -      3.7627572270242035d11*t9113-1.306938363480568d7*t913-
     -      7.759400329627692d10*t9133+
     -      2.2035541165234478d10*t9143+
     -      5.0766172858087546d8*t9163-2.7788602430469297d7*t9173+
     -      2.1772986482753137d8*t923+1.2801286429769983d10*t943-
     -      5.3194075818154396d10*t953-3.2614409743461664d11*t973+
     -      5.0956714683245935d11*t983)*t9m23/ex(t9m13)
	else
	  f(29)=4.16618d7
	  fp=4.20784d7
	  fm=4.12452d7
	endif
	if ((nchrat.ne.0).and.(t9.le.10.d0)) then
	  drate(29,1)=fm/f(29)-1.d0
	  drate(29,2)=fp/f(29)-1.d0
	endif

	elseif(ddp.eq.'PIS2020noTH') then

C.......H2(d,p)H3..................(Pisanti et al 2020, no TH)
	if (t9.le.4.d0) then
	  f(29)=(342120.91162029817d0-2.0884162974120884d9*t9+
     -      1.5500150027173837d11*t9**2-
     -      6.018878745229017d11*t9**3+
     -      1.9743931580254926d11*t9**4-
     -      4.2701353479700108d9*t9**5+
     -      5.4365102307214343d11*t9103-
     -      3.7581952067247784d11*t9113-1.309527272008109d7*t913-
     -      7.748131956932741d10*t9133+2.200133517476577d10*t9143+
     -      5.067843536204927d8*t9163-2.7738373465825792d7*t9173+
     -      2.18082697618206d8*t923+1.2812245763099388d10*t943-
     -      5.321778492713724d10*t953-3.260316437917953d11*t973+
     -      5.092387314390836d11*t983)*t9m23/ex(t9m13)
	  fp=(345542.12074112456d0-2.1093004603814d9*t9+
     -      1.5655151527306128d11*t9**2-
     -      6.079067532579845d11*t9**3+
     -      1.9941370895514175d11*t9**4-4.312836701276607d9*t9**5+
     -      5.4908753329193353d11*t9103-
     -      3.795777158703079d11*t9113-1.322622545328588d7*t913-
     -      7.825613276256987d10*t9133+2.222134852572182d10*t9143+
     -      5.1185219713375205d8*t9163-2.8015757199092343d7*t9173+
     -      2.2026352459297064d8*t923+1.2940368220669876d10*t943-
     -      5.374996277604292d10*t953-3.2929196022602264d11*t973+
     -      5.1433111874638135d11*t983)*t9m23/ex(t9m13)
	  fm=(338699.70251085825d0-2.0675321344528453d9*t9+
     -      1.534514852697676d11*t9**2-5.958689957799038d11*t9**3+
     -      1.9546492264507642d11*t9**4-4.227433994496454d9*t9**5+
     -      5.382145128432868d11*t9103-
     -      3.7206132546693115d11*t9113-1.2964319994139636d7*t913-
     -      7.670650637382088d10*t9133+
     -      2.1781321823061214d10*t9143+5.017165100847144d8*t9163-
     -      2.7460989731171742d7*t9173+2.159018706420903d8*t923+
     -      1.2684123305552046d10*t943-5.26856070781592d10*t953-
     -      3.227713273552929d11*t973+5.0414634412672766d11*t983)*
     -      t9m23/ex(t9m13)
	else
	  f(29)=4.24587d7
	  fp=4.26928d7
	  fm=4.22245d7
	endif
	if ((nchrat.ne.0).and.(t9.le.10.d0)) then
	  drate(29,1)=fm/f(29)-1.d0
	  drate(29,2)=fp/f(29)-1.d0
	endif

	elseif(ddp.eq.'GI2017') then

C.......H2(d,p)H3..................(Gomez Inesta et al 2017)
C....The fit has a precision of better than 0.4%
	if (t9.le.4.d0) then
	  f(29)=-17652.84189486866d0-3.3014989615467854d7*t9-
     -  3.806542252464633d9*t9**2-6.299665151006474d10*t9**3-
     -  2.2153115318416617d11*t9**4-2.0938337056697278d11*t9**5-
     -  5.170277197544064d10*t9**6-2.3351905672377496d9*t9**7+
     -  1.2444129982494043d11*t9112+1.227747066004276d6*t912+
     -  1.4243316881409819d10*t9132+1.7230684376072207d8*t9152+
     -  4.630880772670493d8*t932+1.9337274973079227d10*t952+
     -  1.4022638135524335d11*t972+2.5291614558200958d11*t992
	  fp=1.011d0*f(29)
	  fm=.989d0*f(29)
	else
	  f(29)=4.251d7
	  fp=1.011d0*f(29)
	  fm=.989d0*f(29)
	endif
	if (nchrat.ne.0) then
	  drate(29,1)=fm/f(29)-1.d0
	  drate(29,2)=fp/f(29)-1.d0
	endif

	elseif(ddp.eq.'COC2015') then

C.......H2(d,p)H3..................(Coc et al 2015)
	if (t9.le.4.d0) then
	  f(29)=-7.952295883703868d8*t9+1.1560450340820157d11*t9**2-
     -  8.318029387484115d11*t9**3+6.104563128564998d11*t9**4+
     -  1.3155833992470047d11*t9**4.666666666666667d0-
     -  3.813349757220062d10*t9**5+
     -  7.521029383175717d9*t9**5.333333333333333d0-
     -  9.018092745562425d8*t9**5.666666666666667d0+
     -  4.954818218161273d7*t9**6+9.621811893135598d11*t9103-
     -  8.700309905822389d11*t9113-1.6445311935329642d6*t913-
     -  3.2767383636873834d11*t9133+5.5597241691331364d7*t923+
     -  6.414533119063679d9*t943-3.2918730856105347d10*t953-
     -  2.935147646992956d11*t973+5.619446646220925d11*t983
	  fp=1.02d0*f(29)
	  fm=.98d0*f(29)
	else
	  f(29)=4.2511d7
	  fp=1.02d0*f(29)
	  fm=.98d0*f(29)
	endif
	if (nchrat.ne.0) then
	  drate(29,1)=fm/f(29)-1.d0
	  drate(29,2)=fp/f(29)-1.d0
	endif

	elseif(ddp.eq.'PIS2007') then

C.......H2(d,p)H3..................(Pisanti et al 2007)
	if (t9.le.4.d0) then
	  f(29)=((-5.8523126d6 + 2.3222535d8*t9 - 9.877862d6*t9**2 + 
     -      5.2331507d7*t913 - 1.7022642d8*t923 - 1.1875268d8*t943 + 
     -      5.2922232d7*t953)*t9m23)/ex(1.0676573d0*t9m13)
	  fp=((-5.7455947d6 + 2.2299893d8*t9 - 9.5242666d6*t9**2 + 
     -      5.1106128d7*t913 - 1.651035d8*t923 - 
     -      1.1215042d8*t943 + 5.0522037d7*t953)*t9m23)/
     -      ex(1.04599646d0*t9m13)
	  fm=((-5.8975937d6 + 2.3572024d8*t9 - 1.0002957d7*t9**2 + 
     -      5.2820931d7*t913 - 1.7219803d8*t923 - 
     -      1.2126621d8*t943 + 5.3809211d7*t953)*t9m23)/
     -      ex(1.07535853d0*t9m13)
	else
	  f(29)=4.021d7
	  fp=4.02597d7
	  fm=4.01609d7
	endif
	if ((nchrat.ne.0).and.(t9.le.10.d0)) then
	  drate(29,1)=-dsqrt(.01d0**2+12.3d0*(fm/f(29)-1.d0)**2)
	  drate(29,2)=dsqrt(.01d0**2+12.3d0*(fp/f(29)-1.d0)**2)
	endif

	else
	  write(*,*) 'error in ddp choice'
	  stop
	endif

C.......H3(d,n)He4.................(Serpico et al 2004)
	if (t9.le.2.5d0) then
	  f(30)=6.2265733d8/(ex(0.49711597d0/t9)*t9**0.56785403d0) + 
     -  ex(-0.23309803d0*t9**2 - 1.342742d0*t9m13)*
     -   (-8.1144927d7 + 2.2315324d9*t9 - 2.9439669d9*t9**2 + 
     -     1.8764462d9*t9**3 - 6.0511612d8*t9**4 + 
     -     9.5196576d7*t9**5 - 5.2901086d6*t9**6)*t9m23
	  fp=6.200594d8/(ex(0.49495969d0/t9)*t9**0.56078105d0) + 
     -  ex(-0.23797125d0*t9**2 - 1.3784792d0*t9m13)*
     -   (-8.7018245d7 + 2.4114301d9*t9 - 
     -     3.2227206d9*t9**2 + 2.0779852d9*t9**3 - 
     -     6.7739586d8*t9**4 + 1.0762439d8*t9**5 - 6.0348254d6*t9**6)*
     -   t9m23
	  fm=6.3798186d8/(ex(0.49598246d0/t9)*t9**0.58460934d0) + 
     -  ex(-0.33273637d0*t9**2 - 1.0508793d0*t9m13)*
     -   (-4.0964097d7 + 1.064899d9*t9 - 7.152721d8*t9**2 - 
     -     1.4155217d8*t9**3 + 3.9276243d8*t9**4 - 
     -     1.5817375d8*t9**5 + 2.128034d7*t9**6)*t9m23
	else
	  f(30)=3.40249d8
	  fp=3.41798d8
	  fm=3.38424d8
	endif
	if (nchrat.ne.0) then
	  drate(30,1)=-dsqrt(.0126d0**2+1.4d0*(fm/f(30)-1.d0)**2)
	  drate(30,2)=dsqrt(.0126d0**2+1.4d0*(fp/f(30)-1.d0)**2)
	endif

C.......He3(d,p)He4................(Serpico et al 2004)
	if (t9.le.2.5d0) then
	  f(31)=3.1038385d8/(ex(1.6190981d0/t9)*t9**0.12159455d0) + 
     -     ex(-0.0062340825d0*t9**2 - 1.4540617d0*t9m13)*
     -     (-3.1335916d7 - 6.2051071d8*t9 - 
     -     1.8782248d9*t9**2 + 6.5642773d8*t9**3 + 
     -     1.530887d8*t9**4 - 4.9542138d8*t9103 - 1.770285d8*t9113 + 
     -     1.14185d8*t913 - 2.516526d7*t9133 + 1.7500204d8*t923 - 
     -     1.7513362d9*t943 + 5.2792247d9*t953 - 3.32382d9*t973 + 
     -     2.0346284d9*t983)*t9m23
	  fp=2.7540767d8/(ex(1.7895761d0/t9)*t9**0.42535964d0) + 
     -     ex(-0.011584496d0*t9**2 - 1.7647266d0*t9m13)*
     -     (-4.0539244d7 - 6.8068775d8*t9 + 1.6755542d9*t9**2 + 
     -     1.3327241d9*t9**3 + 2.5284074d8*t9**4 - 
     -     8.0072489d8*t9103 - 3.2332801d8*t9113 + 1.3990258d8*t913 - 
     -     4.0197501d7*t9133 + 2.4121225d8*t923 - 2.3960064d9*t943 + 
     -     5.3331297d9*t953 - 7.7996883d9*t973 + 
     -     3.3487409d9*t983)*t9m23
	  fm=2.7552759d8/(ex(1.5970464d0/t9)*t9**0.0070474065d0) + 
     -     ex(-0.0067819916d0*t9**2 - 2.0484693d0*t9m13)*
     -     (-4.6389646d6 - 3.2264085d9*t9 + 
     -     8.3768817d10*t9**2 + 4.6593422d10*t9**3 - 
     -     5.3027407d9*t9**4 - 5.1730322d10*t9103 + 
     -     2.3630624d10*t9113 - 2.3467142d8*t913 + 
     -     4.8023403d8*t9133 + 1.6138031d9*t923 + 
     -     5.5023454d9*t943 - 2.9668793d10*t953 - 
     -     9.5677252d10*t973 + 2.4498194d10*t983)*t9m23
	else
	  f(31)=1.55167d8
	  fp=1.55567d8
	  fm=1.54638d8
	endif
	if (nchrat.ne.0) then
	  drate(31,1)=-dsqrt(.00299d0**2+3.8d0*(fm/f(31)-1.d0)**2)
	  drate(31,2)=dsqrt(.00299d0**2+3.8d0*(fp/f(31)-1.d0)**2)
	endif

C-----Three particle reactions

C.......He3(He3,2p)He4.............(NACRE 1999)
	f(32) = 5.59d10*t9m23*ex(-12.277d0/t913)*
     .    (1.d0-.135d0*t9+2.54d-2*t9**2-1.29d-3*t9**3)
	if ((nchrat.ne.0).and.(t9.le.10.d0)) then
	  drate(32,1)=-1.d-2-7.34d-2-.102d0*ex(-t9)+3.23d-4*t9
	  drate(32,2)=-drate(32,1)
	endif

C.......Li7(d,na)He4...............(Serpico et al 2004)
	if (t9.le.2.5d0) then
	  f(33) = 1.66d11*t9m23*ex(-10.254d0/t913)+
     .     1.71d6*t9m32*ex(-3.246d0/t9)+1.49d10*t9m32*ex(-4.0894d0/t9)*
     .     (2.57d-2/t9 + 2.6314d0*t9m23-4.1929/t913 -2.1241+4.1136*t913)
	else
	  f(33)=1.37518d9
	endif
	if (nchrat.ne.0) then
	  drate(33,1)=-.5d0
	  drate(33,2)=.5d0
	endif


C.......Be7(d,pa)He4...............(Caughlan-Fowler 1988)
	f(34) = 1.07d12*t9m23*ex(-12.428d0/t913)
	if (nchrat.ne.0) then
	  drate(34,1)=-.9d0
	  drate(34,2)=9.d0
	endif

C-----New reactions not contained in the original Kawano code network

C.......He3(t,g)Li6................(Fukugita-Kajino 1990)
	f(35) = 1.2201d6*t9m23*ex(-7.73436d0/t913)*
     .    (1.d0+5.38722d-2*t913-.214d0*(1.d0+.377d0*t913)*t923+
     .    .2733d0*(1.d0+.959d0*t913)*t943-
     .    1.53d-2*(1.d0+.959d0*t913)*t9**2)*
     .    (1.d0-.213646d0*t923+.136643d0*t943-7.65244d-3*t9**2)
	if (nchrat.ne.0) then
	  drate(35,1)=-.8d0
	  drate(35,2)=4.d0
	endif

C.......Li6(d,n)Be7................(Malaney-Fowler 1989)
	f(36) = 1.48d12*t9m23*ex(-10.135d0/t913)
	if (nchrat.ne.0) then
	  drate(36,1)=-.5d0
	  drate(36,2)=1.d0
	endif

C.......Li6(d,p)Li7................(Malaney-Fowler 1989)
	f(37) = 1.48d12*t9m23*ex(-10.135d0/t913)
	if (nchrat.ne.0) then
	  drate(37,1)=-.5d0
	  drate(37,2)=1.d0
	endif

C.......He3(t,d)He4................(Caughlan-Fowler 1988)
	t9vs = t9/(1.d0+.128d0*t9)
	f(38) = 5.46d9*t9vs**.8333333d0*t9m32*
     .    ex(-7.733d0/(t9vs**.333333d0))
	if (nchrat.ne.0) then
	  drate(38,1)=-.5d0
	  drate(38,2)=1.d0
	endif

C.......H3(t,2n)He4................(Caughlan-Fowler 1988)
	f(39) = 1.67d9*t9m23*ex(-4.872d0/t913)*
     .    (1.d0+8.6d-2*t913-.455d0*t923-.272d0*t9+
     .    .148d0*t943+.225d0*t953)
	if ((nchrat.ne.0).and.(t9.le.10.d0)) then
	  drate(39,1)=-.5d0
	  drate(39,2)=1.d0
	endif

C.......He3(t,np)He4...............(Caughlan-Fowler 1988)
	t9tt = t9/(1.d0+.115d0*t9)
	f(40) = 7.71d9*t9tt**.8333333d0*t9m32*
     .    ex(-7.733d0/(t9tt**.333333d0))
	if ((nchrat.ne.0).and.(t9.le.10.d0)) then
	  drate(40,1)=-.5d0
	  drate(40,2)=1.d0
	endif

	if (irec.eq.40) goto 100

C-----Intermediate network

C-----Temperature factors

      t915  = t9**(.2d0)                  !t9**(1/5)
      t954  = t9**(1.25d0)                !t9**(5/4)
      t9m34 = dsqrt(t9m32)                !t9**(-3/4)
      t9m15 = 1.d0/t915                   !t9**(-1/5)
      t9m54 = 1.d0/t954                   !t9**(-5/4)
      t9a   = t9/(1.d0+t9/15.1d0)         !For reaction 53.
      t9a13 = t9a**(1./3.d0)              !t9a**(1/3)
      t9a56 = t9a**(.8333333d0)           !t9a**(5/6)

C-----New reactions not contained in the original Kawano code network

C.......Li7(t,n)Be9................(Thomas 1993 from BKKW91)
	if (t9.ge.10.d0) then
	  f(41) = 1.656d7
	else
	  f(41) = 2.98d10*t9m23*ex(-11.333d0/t913)*
     .      (1.d0-.122d0*t923+1.32d0/(t943-.127d0*t923+7.42d-2))
	endif
	if (nchrat.ne.0) then
	  drate(41,1)=-.21d0
	  drate(41,2)=.21d0
	endif

C.......Be7(t,p)Be9................(Serpico et al 2004)
	if (t9.ge.10.d0) then
	  f(42) = 5.98d6
	else
	  f(42) = 1.1d0*2.98d10*t9m23*ex(-13.7307d0/t913)*
     .      (1.d0-.122d0*t923+1.32d0/(t943-.127d0*t923+7.42d-2))
	endif
	if (nchrat.ne.0) then
	  drate(42,1)=-.5d0
	  drate(42,2)=1.d0
	endif

C.......Li7(He3,p)Be9..............(Serpico et al 2004)
	if (t9.ge.10.d0) then
	  f(43) = 1.2013d6
	else
	  f(43) = 1.6d0*2.98d10*t9m23*ex(-17.992d0/t913)*
     .      (1.d0-.122d0*t923+1.32d0/(t943-.127d0*t923+7.42d-2))
	endif
	if (nchrat.ne.0) then
	  drate(43,1)=-.5d0
	  drate(43,2)=1.d0
	endif

C-----Neutron, photon reactions

C.......Li7(n,g)Li8................(Wagoner 1969)
	f(44) = 3.144d3+4.26d3*t9m32*ex(-2.576d0/t9)!Thomas89 from WSK
	if (nchrat.ne.0) then
	  drate(44,1)=-.3d0                 !1xFCZI
	  drate(44,2)=.3d0
	endif

C.......B10(n,g)B11................(Wagoner 1969)
	f(45) = 6.62d4

C.......B11(n,g)B12................(Malaney-Fowler 1989)
	f(46) = 7.29d2+2.4d3*t9m32*ex(-.223d0/t9)

C-----Neutron, proton reactions

C.......C11(n,p)B11................(Caughlan-Fowler 1988)
	f(47) = 1.69d8*(1.d0-.048d0*t912+.010d0*t9)

C-----Neutron, alpha reactions

C.......B10(n,a)Li7................(NACRE 1999)
	f(48) = 2.2d7*(1.d0+1.064d0*t9)
	if ((nchrat.ne.0).and.(t9.le.10.d0)) then
	  drate(48,1)=-.10794d0+2.2003d-2*t9-3.51236d-3*t9**2
	  drate(48,2)=-drate(48,1)
	endif

C-----Proton, photon reactions

C.......Be7(p,g)B8.................(NACRE 1999)
	f(49) = 2.61d5*t9m23*ex(-10.264d0/t913)*(1.d0-5.11d-2*t9+
     .    4.68d-2*t9**2-6.6d-3*t9**3+3.12d-4*t9**4)+
     .    2.05d3*t9m32*ex(-7.345/t9)
	if (nchrat.ne.0) then
	  drate(49,1)=-.15d0
	  drate(49,2)=.15d0
	endif

C.......Be9(p,g)B10................(Caughlan-Fowler 1988)
	f(50) = 1.33d7*t9m23*ex(-10.359d0/t913-(t9/.846d0)**2)*
     .    (1.d0+.04d0*t913+1.52d0*t923+.428d0*t9+
     .    2.15d0*t943+1.54d0*t953)+9.64d4*t9m32*ex(-3.445d0/t9)+
     .    2.72d6*t9m32*ex(-10.62d0/t9)

C.......B10(p,g)C11................(NACRE 1999)
	f(51) = t9m23*1.68d6*ex(-t9m13*12.064d0)/((t923-.0273d0)**2+
     .    4.69d-4)*(1.d0+.977d0*t9+1.87d0*t9**2-.272d0*t9**3+
     .    .013d0*t9**4)
	
C.......B11(p,g)C12................(NACRE 1999)
	f(52) = t9m23*4.58d7*ex(-t9m13*12.097d0-(t9/.6)**2)*
     .    (1.d0+.353d0*t9-.842d0*t9**2)+t9m32*6.82d3*
     .    ex(-t9m1*1.738d0)+2.8d4*t9**.104d0*ex(-t9m1*3.892d0)

C.......C11(p,g)N12................(Caughlan-Fowler 1988)
	f(53) = 4.24d4*t9m23*ex(-13.658d0/t913-(t9/1.627d0)**2)*
     .    (1.d0+.031d0*t913+3.11d0*t923+.665d0*t9+
     .    4.61d0*t943+2.50d0*t953)+
     .    8.84d3*t9m32*ex(-7.021d0/t9)

C-----Proton, neutron reactions

C.......B12(p,n)C12................(Wagoner 1969)
	f(54) = 4.02d11*t9m23*ex(-12.12d0/t913)

C-----Proton, alpha reactions

C.......Be9(p,a)Li6................(NACRE 1999)
C	Fit valid for T9>=.002 (T>=.17keV)
	f(55) = 2.11d11*t9m23*ex(-10.361d0/t913-(t9/.4d0)**2)*
     .    (1.d0-.189d0*t9+3.52d1*t9**2)+
     .    5.24d8*t9m32*ex(-3.446d0/t9)+
     .    4.65d8*ex(-4.396d0/t9)/(t9**.293d0)
	if ((nchrat.ne.0).and.(t9.le.10.d0)) then
	  drate(55,1)=-.15236d0-.676965d0*ex(-20.*t9)+.13113d0*t9*ex(-t9)
	  drate(55,2)=9.849d-2+.9084d0*ex(-20.*t9)+1.21d-2*t9-
     .    5.987d-4*t9**2
	endif

C.......B10(p,a)Be7................(NACRE 1999)
	if (t9.ge..8d0) then
	  f(56) = 1.01d10*t9m23*ex(-12.064d0/t913)*
     .      (-1.d0+15.8d0*t9-2.6d0*t9**2+.125d0*t9**3)
	else
	  f(56) = 2.56d10*t9m23*ex(-12.064d0/t913)*
     .      (1.d0+5.95d0*t9+2.92d1*t9**2-3.16d2*t9**3+9.14d2*t9**4-
     .      1.085d3*t9**5+4.65d2*t9**6)/(4.7d-4+(t923-2.6d-2)**2)	
	endif
	if ((nchrat.ne.0).and.(t9.le.10.d0)) then
	  drate(56,1)=-.101d0-.1234d0*t9+1.418d-2*t9**2-5.79d-4*t9**3
	  drate(56,2)=-drate(56,1)
	endif

C.......B12(p,a)Be9................(Wagoner 1969)
	f(57) = 2.01d11*t9m23*ex(-12.12d0/t913)

C-----Alpha, photon reactions

C.......Li6(a,g)B10................(Caughlan-Fowler 1988)
	f(58) = 4.06d6*t9m23*ex(-18.790d0/t913-(t9/1.326d0)**2)*
     .    (1.d0+.022d0*t913+1.54d0*t923+
     .    .239d0*t9+2.2d0*t943+.869d0*t953)+
     .    1.91d3*t9m32*ex(-3.484d0/t9)+
     .    1.01d4*t9m1*ex(-7.269d0/t9)
	if (nchrat.ne.0) then
	  drate(58,1)=-.5d0
	  drate(58,2)=1.d0
	endif

C.......Li7(a,g)B11................(NACRE 1999)
	if (t9.ge.1.21d0) then
	  f(59) = 1.187d3*t9m32*ex(-2.959d0/t9)+
     .      7.945d3*(1.d0+.1466d0*t9-1.273d-2*t9**2)*
     .      ex(-4.922d0/t9)/(t9**2.3d-2)
	else
	  f(59) = 9.72d7*t9m23*ex(-19.163d0/t913-(t9/.4d0)**2)*
     .      (1.d0+2.84d0*t9-7.89d0*t9**2)+
     .      3.35d2*t9m32*ex(-2.959d0/t9)+
     .      1.04d4*ex(-4.922d0/t9)/(t9**2.3d-2)
	endif
	if ((nchrat.ne.0).and.(t9.le.10.d0)) then
	  drate(59,1)=-.2246d0+3.9114d-2*ex(-.5*(t9-6.d0)**2)+
     .      3.16d-2*ex(-5.d0*(t9-.2d0)**2)-.50145d0*ex(-20.d0*t9)
	  drate(59,2)=.248d0+9.6575d-2*ex(-.5*(t9-6.d0)**2)-
     .      5.796d-2*ex(-5.d0*(t9-.2d0)**2)+.7442d0*ex(-20.d0*t9)
	endif

C.......Be7(a,g)C11................(NACRE 1999)
	if (t9.ge.2.d0) then
	  f(60) = 1.41d3*ex(-3.015d0/t9)*t9**.636d0
	else
	  f(60) = 1.29d10*t9m23*ex(-23.214d0/t913-(t9/.8d0)**2)*
     .      (1.d0-6.47d0*t9+19.5d0*t9**2-19.3*t9**3)+
     .      1.25d4*t9m32*ex(-6.498d0/t9)+1.44d5*t9m32*
     .      ex(-10.177d0/t9)+1.63d4*ex(-15.281d0/t9)*(t9**.178d0)
	endif
	if ((nchrat.ne.0).and.(t9.le.10.d0)) then
	  drate(60,1)=-.35187d0+5.31d-2*t9-4.684d-2*t9**2+
     .      6.771d-3*t9**3-3.021d-4*t9**4
	  drate(60,2)=.3298d0+.127d0*t9-3.611d-2*t9**2+
     .      5.3544d-3*t9**3-2.6134d-4*t9**4
	endif

C-----Alpha, proton reactions

C.......B8(a,p)C11.................(Wagoner 1969)
	f(61) = 1.08d15*t9m23*ex(-27.36d0/t913)

C-----Alpha, neutron reactions

C.......Li8(a,n)B11................(Malaney-Fowler 1989)
	f(62) = 8.62d13*t9a56*t9m32*ex(-19.461d0/t9a13)

C.......Be9(a,n)C12................(Caughlan-Fowler 1988)
	f(63) = 4.62d13*t9m23*ex(-23.87d0/t913-(t9/.049d0)**2)*
     .    (1.d0+.017d0*t913+8.57d0*t923+1.05d0*t9+
     .    74.51d0*t943+23.15d0*t953)+7.34d-5*t9m32*ex(-1.184d0/t9)+
     .    2.27d-1*t9m32*ex(-1.834d0/t9)+1.26d5*t9m32*ex(-4.179d0/t9)+
     .    2.4d8*ex(-12.732d0/t9)

C-----Deuterium, neutron and deuterium, proton reactions

C.......Be9(d,n)B10................(original Wagoner code)
	f(64) = 7.16d8*t9m23*ex(6.44d0-12.6d0/t913)

C.......B10(d,p)B11................(original Wagoner code)
	f(65) = 9.53d8*t9m23*ex(7.3d0-14.8d0/t913)

C.......B11(d,n)C12................(original Wagoner code)
	f(66) = 1.41d9*t9m23*ex(7.4d0-14.8d0/t913)

C-----Three particle reactions

C.......He4(an,g)Be9...............(Caughlan-Fowler 1988)
	f(67) = (2.59d-6/((1.d0+.344d0*t9)*t9**2))*ex(-1.062d0/t9)

C.......He4(2a,g)C12...............(Caughlan-Fowler 1988)
	f(68) = 2.79d-8*t9m32*t9m32*ex(-4.4027d0/t9)+
     .    1.35d-8*t9m32*ex(-24.811d0/t9)

C.......Li8(p,na)He4...............(original Wagoner code)
	f(69) = 8.65d9*t9m23*ex(-8.52d0/t913-(t9/2.53d0)**2)+
     .    2.31d9*t9m32*ex(-4.64d0/t9)

C.......B8(n,pa)He4................(original Wagoner code)
	f(70) = 4.02d8

C.......Be9(p,da)He4...............(Caughlan-Fowler 1988)
	f(71) = 2.11d11*t9m23*ex(-10.359d0/t913-(t9/.520d0)**2)*
     .   (1.d0+.04d0*t913+1.09d0*t923+.307d0*t9+3.21d0*t943+2.3d0*t953)+
     .   5.79d8*t9m1*ex(-3.046d0/t9)+8.5d8*t9m34*ex(-5.8d0/t9)

C.......B11(p,2a)He4...............(Caughlan-Fowler 1988)
	f(72) = 2.2d12*t9m23*ex(-12.095d0/t913-(t9/1.644d0)**2)*
     .    (1.d0+.034d0*t913+.14d0*t923+.034d0*t9+
     .    .19d0*t943+.116d0*t953)+4.03d6*t9m32*ex(-1.734d0/t9)+
     .    6.73d9*t9m32*ex(-6.262d0/t9)+3.88d9*t9m1*ex(-14.154d0/t9)

C.......C11(n,2a)He4...............(Wagoner 1969)
	f(73) = 1.58d8

	if (irec.eq.73) goto 100

C-----Complete network

C-----Temperature factors

      t935  = t9**(.6d0)                  !t9**(3/5)
      t965  = t9**(1.2d0)                 !t9**(6/5)
      t938  = t9**(.375d0)                !t9**(3/8)
      t9m65 = 1.d0/t965                   !t9**(-6/5)
      t9a   = t9                          !For reaction 82.
     |/(1.d0+4.78d-2*t9+7.56d-3*t953/(1.d0+4.78d-2*t9)**(2.d0/3.))
      t9a13 = t9a**(1./3.d0)              !t9a**(1/3)
      t9a56 = t9a**(.83333333d0)          !t9a**(5/6)
      t9b   = t9                          !For reaction 84.
     |/(1.d0+7.76d-2*t9+2.64d-2*t953/(1.d0+7.76d-2*t9)**(2.d0/3.))
      t9b13 = t9b**(1./3.d0)              !t9b**(1/3)
      t9b56 = t9b**(.83333333d0)          !t9b**(5/6)

C-----Neutron, photon reactions

C.......C12(n,g)C13................(Wagoner 1969)
	f(74)  = 4.50d+2

C.......C13(n,g)C14................(Wagoner 1969)
	f(75)  = 1.19d+2+2.38d+5*t9m32*ex(-1.67d0/t9)

C.......N14(n,g)N15................(Wagoner 1969)
	f(76)  = 9.94d+3			 

C-----Neutron, proton reactions

C.......N13(n,p)C13................(NACRE 1999)
	f(77) = 1.178d+8*(1.d0+3.36d-1*t9-3.792d-2*t9**2+2.02d-3*t9**3)
C-----Contribution from thermal excited levels (NACRE 1999)
	f(77) = f(77)*(1.d0+1.131d0*ex(-1.2892d+1*t9m1+1.9d-2*t9))

C.......N14(n,p)C14................(Caughlan-Fowler 1988)
	f(78)  = 2.39d+5*(1.d0+.361d0*t912+.502d0*t9)
     |         + 1.112d+8/t912*ex(-4.983d0/t9)

C.......O15(n,p)N15................(NACRE 1999)
	f(79) = 1.158d+8*(1.d0+2.19d-1*t9-2.9d-2*t9**2+1.73d-3*t9**3)
C-----Contribution from thermal excited levels (NACRE 1999)
	f(79) = f(79)*(1.d0+3.87d-1*ex(-26.171d0*t9m1+1.18d-1*t9))

C-----Neutron, alpha reactions

C.......O15(n,a)C12................(Caughlan-Fowler 1988)
	f(80)  = 3.50d+7*(1.d0+.188d0*t912+.015d0*t9)

C-----Proton, photon reactions

C.......C12(p,g)N13................(NACRE 1999)
	f(81) = 2.00d7*t9m23*ex(-13.692d0*t9m13-(t9/.46d0)**2)*
     .    (1.d0+9.89d0*t9-59.8d0*t9**2+266.d0*t9**3)+1.00d5*t9m32*
     .    ex(-4.913d0*t9m1)+4.24d5*t9m32*ex(-21.62d0*t9m1)

C.......C13(p,g)N14................(NACRE 1999)
	f(82) = 9.57d+7*t9m23*ex(-13.720d0*t9m13-t9**2)*(1.d0+3.56d0*t9)
     |         +1.5d+6*t9m32*ex(-5.930d0*t9m1)+6.83d+5*t9**(-8.64d-1)
     |         *ex(-12.057d0*t9m1)
C-----Contribution from thermal excited levels (NACRE 1999)
	f(82) = f(82)*(1.d0-2.070d0*ex(-37.938d0*t9m1))

C.......C14(p,g)N15................(Caughlan-Fowler 1988)
	f(83)  = 6.80d+6*t9m23*ex(-13.741d0/t913-(t9/5.721d0)**2)
     |         *(1.d0+.030d0*t913+.503d0*t923+.107d0*t9
     |         +.213d0*t943+.115d0*t953)
     |         + 5.36d+3*t9m32*ex(-3.811d0/t9)
     |         + 9.82d+4*t9m13*ex(-4.739d0/t9)

C.......N13(p,g)O14................(Caughlan-Fowler 1988)
	f(84)  = 4.04d+7*t9m23*ex(-15.202d0/t913-(t9/1.191d0)**2)
     |         *(1.d0+.027d0*t913-.803d0*t923-.154d0*t9
     |         +5.00d0*t943+2.44d0*t953)
     |         + 2.43d+5*t9m32*ex(-6.348d0/t9)

C.......N14(p,g)O15................(Caughlan-Fowler 1988)
	f(85)  = 4.90d+7*t9m23*ex(-15.228d0/t913-(t9/3.294d0)**2)
     |         *(1.d0+.027d0*t913-.778d0*t923-.149d0*t9
     |         +.261d0*t943+.127d0*t953)
     |         + 2.37d+3*t9m32*ex(-3.011d0/t9)
     |         + 2.19d+4*ex(-12.530d0/t9)

C.......N15(p,g)O16................(Caughlan-Fowler 1988)
	f(86)  = 9.78d+8*t9m23*ex(-15.251d0/t913-(t9/.450d0)**2)
     |         *(1.d0+.027d0*t913+.219d0*t923+.042d0*t9
     |         +6.83d0*t943+3.32d0*t953)
     |         + 1.11d+4*t9m32*ex(-3.328d0/t9)
     |         + 1.49d+4*t9m32*ex(-4.665d0/t9)
     |         + 3.80d+6*t9m32*ex(-11.048d0/t9)

C-----Proton, alpha reactions

C.......N15(p,a)C12................(NACRE 1999)
	if (t9.gt.2.5d0) then
	  f(87) = 4.17d7*t9**.917d0*ex(-3.292d0*t9m1)
	else
	  f(87) = 1.12d12*t9m23*ex(-15.253d0*t9m13-(t9/.28d0)**2)*
     .      (1.d0+4.95d0*t9+143.d0*t9**2)+1.01d8*t9m32*
     .      ex(-3.643d0*t9m1)+1.19d9*t9m32*ex(-7.406d0*t9m1)
	endif

C-----Alpha, photon reactions

C.......C12(a,g)O16................(Caughlan-Fowler 1988)
	f(88)  = 1.04d+8/t9**2*ex(-32.120d0/t913-(t9/3.496d0)**2)
     |         /(1.d0+.0489d0*t9m23)**2
     |         + 1.76d+8/(t9)**2/(1.d0+.2654d0*t9m23)**2
     |         *ex(-32.120d0/t913)
     |         + 1.25d+3*t9m32*ex(-27.499d0/t9)
     |         + 1.43d-2*(t9)**5*ex(-15.541d0/t9)

C-----Alpha, proton reactions

C.......B10(a,p)C13................(Wagoner 1969)
	f(89)  = 9.60d+14*t9m23*ex(-27.99d0/t913)

C.......B11(a,p)C14................(Caughlan-Fowler 1988)
	f(90)  = 5.37d+11*t9m23*ex(-28.234d0/t913-(t9/0.347d0)**2)
     |         *(1.d0+.015d0*t913+5.575d0*t923+.576d0*t9
     |         +15.888d0*t943+4.174d0*t953)
     |         + 5.44d-3*t9m32*ex(-2.827d0/t9)
     |         + 3.36d+2*t9m32*ex(-5.178d0/t9)
     |         + 5.32d+6/t938*ex(-11.617d0/t9)

C.......C11(a,p)N14................(Caughlan-Fowler 1988)
	f(91) = 7.15d+15*t9a56*t9m32*ex(-31.883d0/t9a13)
C-----Contribution from thermal excited levels (NACRE 1999)
	f(91) = f(91)*(1.d0+.140d0*ex(-.275*t9m1-.210d0*t9))

C.......N12(a,p)O15................(Caughlan-Fowler 1988)
	f(92)  = 5.59d+16*t9m23*ex(-35.60d0/t913)

C.......N13(a,p)O16................(Caughlan-Fowler 1988)
	f(93)  = 3.23d+17*t9b56*t9m32*ex(-35.829d0/t9b13)

C-----Alpha, neutron reactions

C.......B10(a,n)N13................(Caughlan-Fowler 1988)
	f(94)  = 1.20d+13*t9m23*ex(-27.989d0/t913-(t9/9.589d0)**2)

C.......B11(a,n)N14................(Caughlan-Fowler 1988)
	f(95)  = 6.97d+12*t9m23*ex(-28.234d0/t913-(t9/0.140d0)**2)
     |         *(1.d0+.015d0*t913+8.115d0*t923+.838d0*t9
     |         +39.804d0*t943+10.456d0*t953)
     |         + 1.79d+0*t9m32*ex(-2.827d0/t9)
     |         + 1.71d+3*t9m32*ex(-5.178d0/t9)
     |         + 4.49d+6*t935*ex(-8.596d0/t9)

C.......B12(a,n)N15................(Wagoner 1969)
	f(96)  = 3.04d+15*t9m23*ex(-28.45d0/t913)

C.......C13(a,n)O16................(NACRE 1999)
	if (t9.ge.4.d0) then
	  f(97) = 7.59d6*t9**1.078d0*ex(-12.056d0*t9m1)
	else
	  f(97) = 3.78d+14*t9m1**2*ex(-32.333d0*t9m13-(t9/.71d0)**2)
     |         *(1.d0+4.68d+1*t9-2.92d+2*t9**2+7.38d+2*t9**3)
     |         +2.3d+7*t9**.45d0*ex(-13.03d0*t9m1)
	endif
C-----Contribution from thermal excited levels (NACRE 1999)
	f(97) = f(97)*(1.d0+7.3318d+1*ex(-58.176d0*t9m1-1.98d-1*t9))

C.......B11(d,p)B12................(Iocco et al 2007)
	f(98)=.60221d0*((4.97838d13*ex(-14.8348d0*(.0215424d0+t9m1)**
     .    (1.d0/3.d0))*(.0215424d0+t9m1)**(1/6.d0)*t9m1**(.5d0))/
     .    (46.42d0+t9)+ex(-14.835d0*t9m13)*(6.52581d9+(1.56921d11-
     .    4.074d9*t913+1.28302d9*t9)*t9m23))
	if (nchrat.ne.0) then
	  drate(98,1)=-.9d0
	  drate(98,2)=9.d0
	endif

C.......C12(d,p)C13................(Iocco et al 2007)
	f(99) = t9m23*6.60999d12*ex(-16.8242d0*t9m13-.234041d0*t923)*
     .    (1.d0-ex(-1.10272*(1.d0+.921391d0*t913)**2))
	if (nchrat.ne.0) then
	  drate(99,1)=-.9d0
	  drate(99,2)=9.d0
	endif

C.......C13(d,p)C14................(Iocco et al 2007)
	f(100) = t9m23*7.23773d12*ex(-16.8869d0*t9m13-.242434d0*t923)*
     .    (1.d0-ex(-1.08715d0*(1.d0+.944456d0*t913)**2))
	if (nchrat.ne.0) then
	  drate(100,1)=-.9d0
	  drate(100,2)=9.d0
	endif

C-----For avoiding negative non physical values
100	do i=1,irec
	  f(i)=dabs(f(i))
	enddo

	if (nchrat.ne.0) then
C-----Rate redefinition
	  do i=2,irec
	    if (hchrat(i).gt.0) then
	      if (hchrat(i).eq.3) then
		f(i)=factor(i)*f(i)
	      else
		f(i)=(1.d0+drate(i,hchrat(i)))*f(i)
	      endif
	    endif
	  enddo
	endif

C----Calculation of nuclear partition functions.
	if (t9.lt.11.d0) then
	  gnpf(0)=1.d0
	  do i=1,inuc
	    gnpf(i)=1.d0+ag(i,1)*ex(ag(i,4)*t9+ag(i,3)/t9)*t9**ag(i,2)
	  enddo
	endif
C-----tht=[(2*pi*mu*k_BT)/(h^2)]^(3/2) in cm^(-3)
	tht=5.94255d33*t932
	do i=2,irec
	  if (rev(i).ne.0.d0) then
	    r(i)=rev(i)*ex(-q9(i)/t9)*f(i)
	    f(i)=f(i)*nbn**(ni(i)+nj(i)+ng(i)-1)
	    r(i)=r(i)*nbn**(nh(i)+nk(i)+nl(i)-1)
	    if (((ti(i).gt.6).or.(tl(i).gt.6)) .and. t9.lt.10.d0)
     .        r(i)=r(i)*(gnpf(ti(i))**ni(i)*gnpf(tj(i))**nj(i)*
     .        gnpf(tg(i))**ng(i)/(gnpf(th(i))**nh(i)*
     .        gnpf(tk(i))**nk(i)*gnpf(tl(i))**nl(i)))
	    if (ni(i)+nj(i)+ng(i)-nh(i)-nk(i)-nl(i).ne.0)
     .        r(i)=r(i)*(tht/(coef(3)/mu))**(ni(i)+nj(i)+ng(i)-
     .        nh(i)-nk(i)-nl(i))
	  endif
	enddo

	RETURN
	END


	SUBROUTINE WRATE(Z,F1,R1)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	This subroutine evaluates weak rates with standard neutrino
C	  distributions
C
C	Called by RATE
C
C	z=dimensionless inverse temperature
C	f1,r1=n->p and p->n weak rates
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	IMPLICIT DOUBLE PRECISION (A-Z)
C--------------------------Local variables------------------------------
	INTEGER          I
	DIMENSION        ZVAL(13)
C-----Physical parameters
	INTEGER          IXIE1,IXIE2
C--------------------------Common variables-----------------------------
	COMMON/MODPAR/   TAU,DNNU,ETAF,RHOLMBD,XIE,XIX

	INTEGER          IXIE,NXIE
	PARAMETER        (NXIE=21)
	DIMENSION        XIEG0(NXIE)
	COMMON/NCHPOT/   XIEG0,IXIE

	DIMENSION        A(13),B(10),DA(12,NXIE),DB(12,NXIE)
	COMMON/WEAKRATE/ A,B,DA,DB,QNP,QNP1,QPN,QPN1
C-----------------------------------------------------------------------

	do i=1,13
	  zval(i)=z**i
	enddo
	ixie1=ixie
	if (ixie.ne.21) then
	  ixie2=ixie+1
	else
C-----The following instruction is only for avoiding division by zero in
C	the final formula
	  ixie2=ixie-1
	endif
	xie1=xieg0(ixie1)
	xie2=xieg0(ixie2)

C-----Forward rate for weak np reaction: lower value
	f1_1 = 1.d0
	df = 0.d0
	do i=1,13
	  f1_1=f1_1+a(i)/zval(i)
	  if ((i.le.10).and.(z.le.5.10998997931d0)) df=df+da(i,ixie1)/
     .      zval(i)
	enddo
	f1_1=(f1_1*ex(-qnp1/z)+df*ex(da(11,ixie1)*zval(1)+da(12,
     .    ixie1)*zval(2))*ex(-qnp*z))/tau
C-----Reverse rate for weak np reaction: lower value
	if(z.le.5.10998997931d0) then
	  r1_1=-.62173d0
	  dr = 0.d0
	  do i=1,10
	    r1_1=r1_1+b(i)/zval(i)
	    if ((i.le.10).and.(z.le.5.10998997931d0)) dr=dr+db(i,ixie1)/
     .        zval(i)
	  enddo
	  r1_1=(r1_1*ex(-qpn1*z)+dr*ex(db(11,ixie1)*zval(1)+db(12,
     .      ixie1)*zval(2))*ex(-qpn*z))/tau
	else
	  r1_1=0.d0
	endif

C-----Forward rate for weak np reaction: higher value
	f1_2 = 1.d0
	df = 0.d0
	do i=1,13
	  f1_2=f1_2+a(i)/zval(i)
	  if ((i.le.10).and.(z.le.5.10998997931d0)) df=df+da(i,ixie2)/
     .      zval(i)
	enddo
	f1_2=(f1_2*ex(-qnp1/z)+df*ex(da(11,ixie2)*zval(1)+da(12,ixie2)*
     .    zval(2))*ex(-qnp*z))/tau
C-----Reverse rate for weak np reaction: higher value
	if(z.le.5.10998997931d0) then
	  r1_2=-.62173d0
	  dr = 0.d0
	  do i=1,10
	    r1_2=r1_2+b(i)/zval(i)
	    if ((i.le.10).and.(z.le.5.10998997931d0)) dr=dr+db(i,ixie2)/
     .        zval(i)
	  enddo
	  r1_2=(r1_2*ex(-qpn1*z)+dr*ex(db(11,ixie2)*zval(1)+db(12,ixie2)*
     .      zval(2))*ex(-qpn*z))/tau
	else
	  r1_2=0.d0
	endif

	f1=f1_1+(f1_2-f1_1)*(xie-xie1)/(xie2-xie1)
	r1=r1_1+(r1_2-r1_1)*(xie-xie1)/(xie2-xie1)

	RETURN
	END


	SUBROUTINE EQSLIN(ICONV,IERROR)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C 	This subroutine solves a linearized system of equation by Gaussian
C	  elimination. It is the same used in the KAWANO 1992 public code
C
C	Called by FCN
C
C	iconv,ierror= inversion flags
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	IMPLICIT DOUBLE PRECISION (A-Z)
C--------------------------Local variables------------------------------
	INTEGER          I,J,K
C-----Network parameters
	INTEGER          NNUC
	PARAMETER        (NNUC=26)
C-----Matrix inversion
	DIMENSION        AMAT0(NNUC,NNUC),X(NNUC)
	INTEGER          ICONV              !Convergence monitor
	INTEGER          IERROR             !Element which does not converge
	INTEGER          MORD               !Higher order in correction
	PARAMETER        (MORD=1)
	INTEGER          NORD               !Order of correction
	PARAMETER        (EPS=2.D-4)        !Tolerance for convergence (.ge.1.d-7)
C--------------------------Common variables-----------------------------
	INTEGER          MBAD
	INTEGER          INC
	COMMON/INVFLAGS/ MBAD,INC

	DIMENSION        AMAT(NNUC,NNUC)
	DIMENSION        BVEC(NNUC)
	DIMENSION        YX(NNUC)
	COMMON/LINCOEF/  AMAT,BVEC,YX

	INTEGER          INUC,IREC,NVXT,IXT(30)
	COMMON/NETWRK/   INUC,IREC,NVXT,IXT
C-----------------------------------------------------------------------

	ierror=0
	nord=0                              !No corrections yet
	mbad=0                              !No errors yet

C..........SET RIGHT-HAND AND SOLUTION VECTORS TO INITIAL VALUES.
	do i=1,inuc
	  x(i)=bvec(i)                      !Right-hand vector
	  yx(i)=0.d0                        !Solution vector
	enddo
C..........SAVE MATRIX.
	if (iconv.eq.inc) then              !Monitor convergence
	  do i=1,inuc
	    do j=1,inuc
	      amat0(i,j) = amat(i,j)        !Initial value of coefficient array
	    enddo
	  enddo
	endif

C20--------TRIANGULARIZE MATRIX AND SAVE OPERATOR--------------------

C..........CHECK TO SEE THAT THERE ARE NO ZEROES AT PIVOT POINTS.
	do i=1,inuc-1
	  if (amat(i,i).eq.0.d0) then       !Don't want to divide by zero
	    mbad=i                          !Position of zero coefficient
	    return                          !Terminate matrix evaluation
	  endif
C..........TRIANGULARIZE MATRIX.
	  do j=i+1,inuc
	    if (amat(j,i).ne.0.d0) then     !Progress diagonally down the column
	      cx=amat(j,i)/amat(i,i)        !Scaling factor down the column
	      do k=i+1,inuc                 !Progress diagonally along row
		amat(j,k)=amat(j,k)-cx*amat(i,k)!Subtract scaled coeff along row
	      enddo
	      amat(j,i)=cx                  !Scaled coefficient
C..........OPERATE ON RIGHT-HAND VECTOR.
	      x(j)=x(j)-cx*x(i)             !Subtract off scaled coefficient
	    endif
	  enddo
	enddo

C30--------DO BACK SUBSTITUTION-------------------------------------------------
300	continue
	x(inuc)=x(inuc)/amat(inuc,inuc)     !Solution for ultimate position
	yx(inuc)=yx(inuc)+x(inuc)
	do i=inuc-1,1,-1                    !From i = penultimate to i = 1
	  sum = 0.d0
	  do j=i+1,inuc
	    sum=sum+amat(i,j)*x(j)          !Sum up all previous terms
	  enddo
	  x(i)=(x(i)-sum)/amat(i,i) 
	  yx(i)=yx(i)+x(i)                  !Add difference to initial value
	enddo

C40--------TESTS AND EXITS------------------------------------------------------

	if (iconv.eq.inc) then
	  do i=1,inuc
	    if (yx(i).ne.0.d0) then
	      xdy=dabs(x(i)/yx(i))          !Relative error
	      if (xdy.gt.eps) then
		if (nord.lt.mord) then          !Continue to higher orders
		  nord=nord+1
C..........FIND ERROR IN RIGHT-HAND VECTOR.
		  do j=1,inuc
		    r=0.d0                      !Initialize r
		    do k=1,inuc
		      r=r+amat0(j,k)*yx(k)      !Left side with approximate solution
		    enddo
		    x(j)=bvec(j)-r              !Subtract difference from right side
		  enddo
C..........OPERATE ON RIGHT-HAND VECTOR.
		  do j=1,inuc-1
		    do k=j+1,inuc
		      x(k)=x(k)-amat(k,j)*x(j)  !Subtract off scaled coefficient
		    enddo
		  enddo
		  go to 300                     !Go for another iteration
		else !(nord.lt.mord)
C..........NOT ENOUGH CONVERGENCE.
		  mbad=-1                       !Signal error problem
		  ierror=i                      !ith nuclide for which x/y checked
		  return
		endif
	      endif
	    endif
	  enddo
	endif

	RETURN
	END


	DOUBLE PRECISION FUNCTION EX(X)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	Exponential function with underflow precaution.
C
C	Called by several subroutines
C
C	x=argument
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	IMPLICIT DOUBLE PRECISION (A-Z)

C-----For compatibility with old VAX/VMS systems
	if (x.gt.88.029d0) then        !In danger of overflow.
	  ex = dexp(88.029d0)
	else
	  if (x.lt.-88.722d0) then     !In danger of underflow.
	    ex = 0.d0
	  else                         !Value of x in allowed range.
	    ex = dexp(x)
	  endif
	endif

	RETURN
	END


	INTEGER FUNCTION FACT(NUMB)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	Factorial function
C
C	Called by FCN
C
C	numb=argument
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	INTEGER          NUMB,I

	fact=1
	if (numb.gt.1) then
	  do i=1,numb
	    fact=i*fact
	  enddo
	endif

	RETURN
	END


	SUBROUTINE OUTEVOL(ZEND,YY)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	This subroutine can print the actual values of the selected
C	  nuclide abundances versus z in one of the output files and the
C	  status of the evolution on the screen
C
C	Called by FCN
C
C	zend=final value of z
C	yy=array of unknown functions
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	IMPLICIT DOUBLE PRECISION (A-Z)
C--------------------------Local variables------------------------------
	INTEGER          I,J,NCOUNT
	DATA             NCOUNT/23/
C-----Network parameters
	INTEGER          NNUC
	PARAMETER        (NNUC=26)
C-----Differential equation resolution parameters
	DIMENSION        YY(NNUC+1)
C--------------------------Common variables-----------------------------
	DIMENSION        COEF(4)
	EQUIVALENCE      (ALP,COEF(1)),(BET,COEF(2)),(GAM,COEF(3)),
     .	             (DEL,COEF(4))
	COMMON/CONSTANTS/PI,COEF,ME,MU,QVAL,ALF,GN

	INTEGER          IFCN,IOUTEV
	COMMON/COUNTS/   IFCN,IOUTEV

	CHARACTER        CMODE*1
	LOGICAL          FOLLOW,OVERW
	COMMON/INPCARD/  FOLLOW,OVERW,CMODE

	COMMON/MODPAR/   TAU,DNNU,ETAF,RHOLMBD,XIE,XIX

	INTEGER          INUC,IREC,NVXT,IXT(30)
	COMMON/NETWRK/   INUC,IREC,NVXT,IXT

	CHARACTER*500    NAMEFILE1,NAMEFILE2,NAMEFILE3
	COMMON/OUTFILES/ NAMEFILE1,NAMEFILE2,NAMEFILE3

	INTEGER          NOUT
	DIMENSION        OUTVAR(20)
	CHARACTER*7      OUTTXT(20)
	COMMON/OUTPUTS/  OUTVAR,OUTTXT,NOUT

	DIMENSION        YYOLD(3,NNUC+1),TMPVAR(3,5)
	COMMON/YOLD/     YYOLD,TMPVAR
C-----------------------------------------------------------------------

C-----Counter increment
	ioutev=ioutev+1

	if (ioutev.lt.3) then
	  tmpvar(ioutev,1)=me/zend
	  tmpvar(ioutev,2)=yy(1)
	  do i=1,nout
	    tmpvar(ioutev,2+i)=outvar(i)
	  enddo
	  do i=1,inuc+1
	    yyold(ioutev,i)=yy(i)
	  enddo
	  if (ixt(30).eq.1 .and. ioutev.eq.2) write(3,16) (tmpvar(1,i),
     .      i=1,nout+2),(yyold(1,ixt(i)+1),i=1,nvxt)
	else
C-----We make a check on the smoothness of the abundances for cleaning
C	some numerical spikes from data
	  tmpvar(3,1)=me/zend
	  tmpvar(3,2)=yy(1)
	  do i=1,nout
	    tmpvar(3,2+i)=outvar(i)
	  enddo
	  do i=1,inuc+1
	    yyold(3,i)=yy(i)
	  enddo
C-----We make the check only for the nuclides to be followed
	  do i=1,nvxt
	    diff1=dabs(yyold(1,ixt(i)+1)-yyold(2,ixt(i)+1))
	    diff2=dabs(yyold(1,ixt(i)+1)-yyold(3,ixt(i)+1))
	    if (diff2.eq.0.d0) then
	      if (diff1.ne.0.d0) goto 1
	    else
	      if (diff1/diff2.gt.5.d1) goto 1
	    endif
	  enddo
	  if (follow) then
	    ncount=ncount+1
	    if (ncount.eq.24) then
	      write(*,*) '   N_nu    xie    xix  tau  rholmbd  eta10',
     .          '     T(MeV)         Y_D         Y_He4        Y_Li7'
	      ncount=0
	    endif
	    write(*,15) 3.d0+dnnu,xie,xix,tau,rholmbd,etaf*1.d10,
     .        tmpvar(2,1),yyold(2,4),yyold(2,7),yyold(2,9)
	  endif
	  if (ixt(30).eq.1) write(3,16) (tmpvar(2,i),i=1,nout+2),
     .      (yyold(2,ixt(i)+1),i=1,nvxt)
	  do j=1,2
	    do i=1,nout+2
	      tmpvar(j,i)=tmpvar(j+1,i)
	    enddo
	    do i=1,inuc+1
	      yyold(j,i)=yyold(j+1,i)
	    enddo
	  enddo
	  return
1	  do i=1,nout+2
	    tmpvar(2,i)=tmpvar(3,i)
	  enddo
	  do i=1,inuc+1
	    yyold(2,i)=yyold(3,i)
	  enddo
	endif

15	format(1x,5f7.2,f8.5,4e13.5)
16	format(1x,d12.7,48d15.7)

	RETURN
	END


	SUBROUTINE OUTEND(ZEND,YY)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C	This subroutine prints the final values of all nuclide abundances
C	  in one of the output files
C
C	Called by PARTHENOPE
C
C	zend=final value of z
C	yy=array of unknown functions
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	IMPLICIT DOUBLE PRECISION (A-Z)
C--------------------------Local variables------------------------------
	INTEGER          I
C-----Network parameters
	INTEGER          NNUC
	PARAMETER        (NNUC=26)
C-----Differential equation resolution parameters
	DIMENSION        YY(NNUC+1)
C-----Final abundances
	DIMENSION        ABUND(NNUC)
C--------------------------Common variables-----------------------------
	INTEGER          AA(NNUC)
	COMMON/ANUM/     AA

	DIMENSION        COEF(4)
	EQUIVALENCE      (ALP,COEF(1)),(BET,COEF(2)),(GAM,COEF(3)),
     .	             (DEL,COEF(4))
	COMMON/CONSTANTS/PI,COEF,ME,MU,QVAL,ALF,GN

	DIMENSION        DM(0:NNUC)
	COMMON/DMASS/    DM

	DIMENSION        YY0(NNUC+1)
	COMMON/INABUN/   YY0

	COMMON/MODPAR/   TAU,DNNU,ETAF,RHOLMBD,XIE,XIX

	INTEGER          INUC,IREC,NVXT,IXT(30)
	COMMON/NETWRK/   INUC,IREC,NVXT,IXT

	CHARACTER*5      BYY(NNUC)
	CHARACTER*5      CYY(NNUC)
	COMMON/NSYMB/    BYY,CYY

	CHARACTER*500    NAMEFILE1,NAMEFILE2,NAMEFILE3
	COMMON/OUTFILES/ NAMEFILE1,NAMEFILE2,NAMEFILE3

	INTEGER          NOUT
	DIMENSION        OUTVAR(20)
	CHARACTER*7      OUTTXT(20)
	COMMON/OUTPUTS/  OUTVAR,OUTTXT,NOUT
C-----------------------------------------------------------------------

C-----Final output
	do i=1,inuc
	  abund(i)=yy(i+1)/yy(3)
	enddo
	hydrogen=(aa(2)+dm(2)/mu)*yy(3)/((aa(1)+dm(1)/mu)*yy(2)+
     .        (aa(2)+dm(2)/mu)*yy(3)+(aa(6)+dm(6)/mu)*yy(7))
	helium3=(yy(6)+yy(5))/yy(3)
	yp=aa(6)*yy(7)
	lithium=(yy(9)+yy(10))/yy(3)
	abund(2)=hydrogen
	abund(5)=helium3
	abund(6)=yp
	abund(8)=lithium
	write(4,'(2(4x,a,f8.5))')
     .    'eta10 = ',etaf*1.d10,'Omega_B h^2 = ',etaf*1.d10*
     .    (1.d0-0.007125*yp)/273.279
	write(4,*)
	if (ixt(30).eq.1) then
	  write(3,16) me/zend,yy(1),(outvar(i),i=1,nout),
     .      (yy(ixt(i)+1),i=1,nvxt)
	endif
	write(2,17) 3.d0+dnnu,xie,xix,tau,rholmbd,etaf*1.d10,
     .    etaf*1.d10*(1.d0-0.007125*yp)/273.279,yy(1),
     .    (outvar(i),i=1,nout),(abund(ixt(i)),i=1,nvxt)

14	format(13x,a,2x,2(d12.5,a))
15	format(11x,2a,2x,2(d12.5,a))
16	format(1x,d10.5,48d13.5)
17	format(1x,f6.2,2f7.2,2f8.2,2f9.5,48d14.6)

	RETURN
	END


	BLOCK DATA
	IMPLICIT DOUBLE PRECISION (A-Z)
C--------------------------Local variables------------------------------
	INTEGER          I,J
C-----Network parameters
	INTEGER          NNUC,NREC
	PARAMETER        (NNUC=26,NREC=100)
C--------------------------Common variables-----------------------------
	INTEGER          AA(NNUC)
	COMMON/ANUM/     AA

	DIMENSION        COEF(4)
	EQUIVALENCE      (ALP,COEF(1)),(BET,COEF(2)),(GAM,COEF(3)),
     .	             (DEL,COEF(4))
	COMMON/CONSTANTS/PI,COEF,ME,MU,QVAL,ALF,GN

	COMMON/DELTAZ/   DZ0,DZ

	DIMENSION        DM(0:NNUC)
	COMMON/DMASS/    DM

	DIMENSION        AG(NNUC,4)
	COMMON/GPART/    AG

	INTEGER          IXIE,NXIE
	PARAMETER        (NXIE=21)
	DIMENSION        XIEG0(NXIE)
	COMMON/NCHPOT/   XIEG0,IXIE

	COMMON/MINABUN/  YMIN

	CHARACTER*5      BYY(NNUC)
	CHARACTER*5      CYY(NNUC)
	COMMON/NSYMB/    BYY,CYY

	COMMON/NUCMASS/  MN,MP

	INTEGER          RATEPAR(NREC,13)
	COMMON/RECPAR0/  RATEPAR

	CHARACTER*14     RSTRING(NREC)
	COMMON/RSTRINGS/ RSTRING

	DIMENSION        GNUC(NNUC)
	COMMON/SPINDF/   GNUC

	DIMENSION        A(13),B(10),DA(12,NXIE),DB(12,NXIE)
	COMMON/WEAKRATE/ A,B,DA,DB,QNP,QNP1,QPN,QPN1

	INTEGER          ZZ(NNUC)
	COMMON/ZNUM/     ZZ
C-----------------------------------------------------------------------

	data pi/3.14159265359d0/
C-----Unity conversion factors (alp,bet,gam,del)
C		1 sec=1.519266889897012 10^21 MeV^(-1):=alp MeV^(-1)
C		1 cm=5.067727095383446 10^10 MeV^(-1):=bet MeV^(-1)
C		1 gr=5.609586137471903 10^26 MeV:=gam MeV
C		K=.08617385341049728 10^(-9) MeV:=del 10^(-9) MeV
	data coef/1.519266889897012d21,5.067727095383446d10,
     .    5.609586137471903d26,.08617385341049728d0/
C-----Masses: electron, proton, neutron and amu in MeV
	data me/.5109990615d0/
	data mp/938.27231d0/
	data mn/939.56563d0/
	data mu/931.4943228d0/
C-----Dimensionless mass difference between neutron and proton
	data qval/2.530963552464583d0/
C-----Fine structure constant
	data alf/0.007299270072993d0/
C-----Newton constant in MeV^(-2). Quoted value is [RPP2006]
C	The uncertainty is +-0.0010d-45
	data gn/6.7087d-45/
C-----Grid values of the nu_e chemical potential
	data xieg0/-1.d0,-.9d0,-.8d0,-.7d0,-.6d0,-.5d0,-.4d0,-.3d0,-.2d0,
     .    -.1d0,0.d0,.1d0,.2d0,.3d0,.4d0,.5d0,.6d0,.7d0,.8d0,.9d0,
     .    1.d0/
C-----Starting value for dz (used for the calculation of derivatives)
	data dz0/1.d-7/
C-----Numerical zero of abundances
	data ymin/1.d-30/
C-----Text strings for the output
	data byy/'    N','    P','   H2','   H3','  He3',
     .    '  He4','  Li6','  Li7','  Be7','  Li8','   B8','  Be9',
     .    '  B10','  B11','  C11','  B12','  C12','  N12','  C13',
     .    '  N13','  C14','  N14','  O14','  N15','  O15','  O16'/
	data cyy/'  N/H','  Y_H',' H2/H',' H3/H','He3/H',
     .    '  Y_p','Li6/H','Li7/H','Be7/H','Li8/H',' B8/H','Be9/H',
     .    'B10/H','B11/H','C11/H','B12/H','C12/H','N12/H','C13/H',
     .    'N13/H','C14/H','N14/H','O14/H','N15/H','O15/H','O16/H'/
C-----Nuclide numbering
C    --------------------------------
C    1) N         7) Li6      13) B10      19) C13      25) O15
C    2) P         8) Li7      14) B11      20) N13      26) O16
C    3) H2        9) Be7      15) C11      21) C14
C    4) H3       10) Li8      16) B12      22) N14
C    5) He3      11) B8       17) C12      23) O14
C    6) He4      12) Be9      18) N12      24) N15
C-----Nuclide atomic numbers
      DATA aa/1,1,2,3,3,4,6,7,7,8,8,9,10,11,11,12,12,12,13,13,14,14,14,
     .  15,15,16/
C-----Nuclide atomic charges
      DATA zz/0,1,1,1,2,2,3,3,4,3,5,4,5,5,6,5,6,7,6,7,6,7,8,7,8,8/
C-----Nuclide mass excesses in MeV (Audi e Wapstra 1997)
	data (dm(i),i=0,nnuc)/0.d0,8.071388d0,7.289028d0,13.135825d0,
     .14.949915d0,14.931325d0,2.424931d0,14.0864d0,14.9078d0,
     .15.7696d0,20.9464d0,22.9212d0,11.34758d0,12.05086d0,
     .8.6680d0,10.6506d0,13.3690d0,0.d0,17.3382d0,3.125036d0,
     .5.3455d0,3.019916d0,2.863440d0,8.006521d0,.101439d0,
     .2.8554d0,-4.737036d0/
C-----Nuclide spin degrees of freedom.
	DATA (gnuc(i),i=1,nnuc)/2.d0,2.d0,3.d0,2.d0,2.d0,1.d0,3.d0,
     .	4.d0,4.d0,5.d0,5.d0,4.d0,7.d0,4.d0,4.d0,3.d0,1.d0,
     .	3.d0,2.d0,2.d0,1.d0,3.d0,1.d0,2.d0,2.d0,1.d0/
C-----Coefficients for the nuclear partition functions G(i), normalized
C	to the fundamental state multiplicity
	DATA ((ag(i,j),j=1,4),i=1,nnuc) /
C	    a1       a2        a3       a4    
     |  0.0d0,   0.0d0,    0.0d0,   0.0d0,        !n
     |  0.0d0,   0.0d0,    0.0d0,   0.0d0,        !p
     |  0.0d0,   0.0d0,    0.0d0,   0.0d0,        !H2
     |  0.0d0,   0.0d0,    0.0d0,   0.0d0,        !H3
     |  0.0d0,   0.0d0,    0.0d0,   0.0d0,        !He3
     |  0.0d0,   0.0d0,    0.0d0,   0.0d0,        !He4
     |  3.47d0, -3.43d-1, -2.59d1,  5.58d-2,      !Li6
     |  5.20d-1,-5.75d-2, -5.59d0,  1.27d-2,      !Li7
     |  5.16d-1,-5.18d-2, -5.02d0,  1.20d-2,      !Be7
     |  0.0d0,   0.0d0,    0.0d0,   0.0d0,        !Li8
     |  0.0d0,   0.0d0,    0.0d0,   0.0d0,        !B8
     |  1.73d-2, 2.25d0,  -1.54d1, -1.08d-1,      !Be9
     |  5.04d-1,-3.19d-1, -8.59d0,  9.40d-2,      !B10
     |  2.16d0, -1.19d0,  -2.67d1,  1.74d-1,      !B11
     |  2.89d0, -1.38d0,  -2.57d1,  1.95d-1,      !C11
     |  0.0d0,   0.0d0,    0.0d0,   0.0d0,        !B12
     |  4.55d0,  1.64d-2, -5.13d1,  3.91d-3,      !C12
     |  0.0d0,   0.0d0,    0.0d0,   0.0d0,        !N12
     |  4.44d-1, 1.06d0 ,  -3.57d1, -4.57d-2,     !C13
     |  4.68d-2, 1.69d0 ,  -2.33d1, -3.84d-2,     !N13
     |  0.0d0,   0.0d0  ,    0.0d0,   0.0d0,      !C14
     |  1.23d0, -1.48d0 ,  -2.85d1,  2.97d-1,     !N14
     |  3.16d-1, 1.38d0 ,  -5.75d1, -4.25d-2,     !O14
     |  1.61d0,  1.60d-1, -5.94d1,  7.03d-2,      !N15
     |  2.95d0, -3.40d-2, -6.04d1,  7.82d-2,      !O15
     |  2.38d0,  5.15d-1, -6.87d1,  1.53d-2/      !O16
C-----Weak rate fit coefficients
	data a/.15735d0,.46172d1,-.40520d2,.13875d3,-.59898d2,.66752d2,
     .    -.16705d2,.38071d1,-.39140d0,.23590d-1,-.83696d-4,-.42095d-4,
     .    .17675d-5/
	data b/.22211d2,-.72798d2,.11571d3,-.11763d2,.45521d2,-.37973d1,
     .    .41266d0,-.26210d-1,.87934d-3,-.12016d-4/
	data qnp/.340994d0/
	data qnp1/.33979d0/
	data qpn/2.89858d0/
	data qpn1/2.8602d0/
	data da /-0.5756538509d+01,0.6624650134d+01,-0.2779356389d+02,
     .    -0.1006522250d+02,-0.1117797846d+02,0.4579948835d+00,
     .    -0.4442017939d-01,0.2394326141d-02,-0.6703332245d-04,
     .    0.7578389661d-06,-0.5463508568d+00, 0.4192034755d-01,   !-1.
     .    -0.3823902484d+01,0.3481772250d+01,-0.2029615335d+02,
     .    -0.1152153012d+02,-0.9464935821d+01,0.2326464807d+00,
     .    -0.1891555554d-01,0.7546108909d-03,-0.1160614985d-04,
     .    -0.5018118733d-08,-0.4624041240d+00, 0.3431536540d-01,  !-.9
     .    -0.2622626828d+01,0.1646143051d+01,-0.1528541704d+02,
     .    -0.1191153818d+02,-0.8120586342d+01,0.8798549678d-01,
     .    -0.2502291601d-02,-0.3008679113d-03,0.2405185600d-04,
     .    -0.4953989238d-06,-0.4011086259d+00, 0.2930185964d-01,  !-.8
     .    0.2281275916d+00,-0.9093590433d+00,-0.1758826852d+01,
     .    -0.1295374219d+02,-0.5594056914d+01,-0.3001725551d+00,
     .    0.4391829279d-01,-0.3434145044d-02,0.1345275800d-03,
     .    -0.2072033776d-05,-0.4974891799d-04, 0.6818821172d-01,  !-.7
     .    -0.3638122498d+01,0.3044014599d+01,-0.1953976250d+02,
     .    -0.7534093717d+01,-0.7840528030d+01,0.3346796493d+00,
     .    -0.3359788805d-01,0.1883845257d-02,-0.5524013010d-04,
     .    0.6596663454d-06,-0.5965606839d+00,0.4717064086d-01,    !-.6
     .    -0.5792631743d+01,0.5320768728d+01,-0.2510152203d+02,
     .    -0.4394163579d+01,-0.7891979459d+01,0.4944064478d+00,
     .    -0.5143505318d-01,0.3026805075d-02,-0.9389258309d-04,
     .    0.1192471678d-05,-0.8197951181d+00,0.6887676164d-01,    !-.5
     .    -0.7807000591d+01,0.8456916818d+01,-0.2950888064d+02,
     .    -0.3263707366d+00,-0.7955311153d+01,0.6758796785d+00,
     .    -0.7230366620d-01,0.4397830022d-02,-0.1412326345d-03,
     .    0.1856400859d-05,-0.9935873887d+00,0.8566443583d-01,    !-.4
     .    -0.4321600553d+01,0.3711380722d+01,-0.1806174542d+02,
     .    -0.2427454463d+01,-0.5401781361d+01,0.3729459313d+00,
     .    -0.3936496633d-01,0.2353756442d-02,-0.7422229009d-04,
     .    0.9581081991d-06,-0.8942454178d+00,0.7580484512d-01,    !-.3
     .    -0.6991167095d+00,-0.3364151274d+00,-0.4951351231d+01,
     .    -0.4425512426d+01,-0.2477852205d+01,0.1697308257d-02,
     .    0.2494445067d-02,-0.3263671738d-03,0.1602448645d-04,
     .    -0.2797415611d-06,-0.5147998599d+00,0.4286480061d-01,   !-.2
     .    -0.2603805051d+00,0.2374392649d-01,-0.2427727063d+01,
     .    -0.1955468683d+01,-0.1396557484d+01,0.2687791621d-01,
     .    -0.2259646587d-02,0.9381108217d-04,-0.1598067743d-05,
     .    0.3358841773d-08,-0.3841016003d+00,0.2957047094d-01,    !-.1
     .    0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,!0.
     .    0.8206647950d+00,-0.2179713726d-01,0.4511925631d+01,
     .    0.2120201483d+01,0.1690630479d+01,-0.5830650298d-01,
     .    0.5455288771d-02,-0.2775282105d-03,0.7148148405d-05,
     .    -0.7173862447d-07,-0.7407240809d+00,0.6192835567d-01,   !.1
     .    0.2048423510d+01,-0.1537666227d+00,0.1042729736d+02,
     .    0.4265147247d+01,0.3666782769d+01,-0.1465228199d+00,
     .    0.1419976668d-01,-0.7604863764d-03,0.2105723554d-04,
     .    -0.2343454979d-06,-0.8042246071d+00,0.6759665363d-01,   !.2
     .    0.4814203065d+01,-0.2091358846d+01,0.2155766853d+02,
     .    0.4527320351d+01,0.6787184602d+01,-0.4479605402d+00,
     .    0.4889885393d-01,-0.3021736165d-02,0.9844049020d-04,
     .    -0.1311888675d-05,-0.9212947296d+00,0.7761309415d-01,   !.3
     .    0.4614713558d+01,0.2781737242d+00,0.2322874939d+02,
     .    0.9666892686d+01,0.8046914303d+01,-0.3180999264d+00,
     .    0.3075873757d-01,-0.1641004063d-02,0.4518865903d-04,
     .    -0.4991434119d-06,-0.8375900694d+00,0.7058742505d-01,   !.4
     .    0.9611817439d+01,-0.2769924449d+01,0.3985081287d+02,
     .    0.9770762583d+01,0.1193133878d+02,-0.6821683507d+00,
     .    0.7009281906d-01,-0.4057071810d-02,0.1234977700d-03,
     .    -0.1537182912d-05,-0.9769853002d+00,0.8293835301d-01,   !.5
     .    0.9043541712d+01,0.3108792222d+00,0.4204244636d+02,
     .    0.1544897764d+02,0.1374338296d+02,-0.6086660458d+00,
     .    0.6015675118d-01,-0.3308159208d-02,0.9474091602d-04,
     .    -0.1099898396d-05,-0.9066929574d+00,0.7661792994d-01,   !.6
     .    0.1952376098d+02,-0.8415420013d+01,0.7282721042d+02,
     .    0.1142802867d+02,0.2008011100d+02,-0.1364492147d+01,
     .    0.1427672710d+00,-0.8454932347d-02,0.2639807578d-03,
     .    -0.3374051497d-05,-0.1075097065d+01,0.9167738949d-01,   !.7
     .    0.3062234982d+02,-0.1850640041d+02,0.1043255520d+03,
     .    0.7295388023d+01,0.2667084823d+02,-0.2114591440d+01,
     .    0.2236610897d+00,-0.1343590367d-01,0.4261029707d-03,
     .    -0.5533331868d-05,-0.1163844210d+01,0.9986616228d-01,   !.8
     .    0.1947418548d+02,-0.5832921183d+00,0.8215441448d+02,
     .    0.2516666279d+02,0.2511345795d+02,-0.1276445238d+01,
     .    0.1290410264d+00,-0.7312006421d-02,0.2171523366d-03,
     .   -0.2630433323d-05,-0.9920973537d+00,0.8393661681d-01,   !.9
     .    0.4104911550d+02,-0.2160974313d+02,0.1399243663d+03,
     .    0.1285967302d+02,0.3605900241d+02,-0.2757179319d+01,
     .    0.2907000018d+00,-0.1739495524d-01,0.5493699482d-03,
     .    -0.7104516945d-05,-0.1168900969d+01,0.9998370960d-01/   !1.
	data db /0.3028183834d+02,-0.1243787866d+03,0.1817641728d+03,
     .    -0.6990226147d+02,0.5842858296d+02,-0.7411275602d+01,
     .    0.8457637691d+00,-0.5473241052d-01,0.1860179406d-02,
     .    -0.2571955630d-04,0.5508833195d-01,-0.9999716345d-02,   !-1.
     .    0.2626207083d+02,-0.1078616352d+03,0.1585227100d+03,
     .    -0.6280893607d+02,0.5136743821d+02,-0.6688943565d+01,
     .    0.7705997962d+00,-0.5028436574d-01,0.1721243838d-02,
     .    -0.2394373345d-04,0.5172247174d-01,-0.9999947468d-02,   !-.9
     .    0.2163381040d+02,-0.8797363351d+02,0.1280085636d+03,
     .    -0.4838215525d+02,0.4132460263d+02,-0.5169262644d+01,
     .    0.5884378316d+00,-0.3799402411d-01,0.1288747872d-02,
     .    -0.1778845278d-04,0.5435494718d-01,-0.9999997527d-02,   !-.8
     .    0.1797151908d+02,-0.7282246114d+02,0.1059518327d+03,
     .    -0.4005386148d+02,0.3433043103d+02,-0.4299782083d+01,
     .    0.4903150432d+00,-0.3170888865d-01,0.1077083751d-02,
     .    -0.1488544098d-04,0.5352347855d-01,-0.9999843587d-02,   !-.7
     .    0.1458538716d+02,-0.5885486230d+02,0.8551927688d+02,
     .    -0.3211482852d+02,0.2777715743d+02,-0.3462230272d+01,
     .    0.3946198842d+00,-0.2550960108d-01,0.8662013720d-03,
     .    -0.1196749575d-04,0.5323253262d-01,-0.9999981296d-02,   !-.6
     .    0.1156294247d+02,-0.4652367753d+02,0.6763518561d+02,
     .    -0.2544085573d+02,0.2203958068d+02,-0.2753026426d+01,
     .    0.3142951019d+00,-0.2034600841d-01,0.6917132773d-03,
     .    -0.9566788133d-05,0.5237394770d-01,-0.9999999799d-02,   !-.5
     .    0.8740026049d+01,-0.3500313157d+02,0.5077090496d+02,
     .    -0.1884150347d+02,0.1656741701d+02,-0.2048098685d+01,
     .    0.2332889174d+00,-0.1507191760d-01,0.5115298723d-03,
     .    -0.7064462611d-05,0.5269172455d-01,-0.9999790895d-02,   !-.4
     .    0.6220298808d+01,-0.2482620109d+02,0.3598840238d+02,
     .    -0.1327949448d+02,0.1177113241d+02,-0.1449782520d+01,
     .    0.1651034647d+00,-0.1066458971d-01,0.3618863101d-03,
     .    -0.4997121172d-05,0.5250024953d-01,-0.9999503808d-02,   !-.3
     .    0.3910506423d+01,-0.1552358850d+02,0.2241910661d+02,
     .    -0.8084277639d+01,0.7328952012d+01,-0.8860172176d+00,
     .    0.1003590262d+00,-0.6451112119d-02,0.2179765128d-03,
     .    -0.2998832889d-05,0.5335233379d-01,-0.9999965780d-02,   !-.2
     .    0.1835552174d+01,-0.7238037489d+01,0.1039189443d+02,
     .    -0.3611588333d+01,0.3388262569d+01,-0.3969323312d+00,
     .    0.4447506317d-01,-0.2830031500d-02,0.9475059096d-04,
     .    -0.1292929160d-05,0.5500763630d-01,-0.9999995009d-02,   !-.1
     .    0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,!0.
     .    -0.1725448919d+01,0.6844820901d+01,-0.1000305393d+02,
     .    0.3778519898d+01,-0.3327155867d+01,0.4216377399d+00,
     .    -0.4877787933d-01,0.3197011434d-02,-0.1099158224d-03,
     .    0.1535434063d-05,0.4941151254d-01,-0.9999596468d-02,    !.1
     .    -0.3189936975d+01,0.1249969134d+02,-0.1799371020d+02,
     .    0.6225569721d+01,-0.5890763264d+01,0.6863975547d+00,
     .    -0.7652853074d-01,0.4835951135d-02,-0.1605792523d-03,
     .    0.2171620862d-05,0.5361889859d-01,-0.9999844490d-02,    !.2
     .    -0.4674166900d+01,0.1841572049d+02,-0.2684281137d+02,
     .    0.9869447162d+01,-0.8919276726d+01,0.1104839390d+01,
     .    -0.1267721339d+00,0.8243784508d-02,-0.2813826906d-03,
     .    0.3905177026d-05,0.5011813395d-01,-0.9999956794d-02,    !.3
     .    -0.5855161223d+01,0.2291368304d+02,-0.3319765725d+02,
     .    0.1170934945d+02,-0.1100272502d+02,0.1319126816d+01,
     .    -0.1498883673d+00,0.9662257149d-02,-0.3273006091d-03,
     .    0.4512800772d-05,0.5235590748d-01,-0.9999516420d-02,    !.4
     .    -0.7056784484d+01,0.2763695214d+02,-0.4023408887d+02,
     .    0.1444200340d+02,-0.1339328036d+02,0.1632349610d+01,
     .    -0.1866421695d+00,0.1209910101d-01,-0.4118585761d-03,
     .    0.5702778238d-05,0.5094819003d-01,-0.9999953614d-02,    !.5
     .    -0.8020576769d+01,0.3126179086d+02,-0.4535999796d+02,
     .    0.1581824622d+02,-0.1507237680d+02,0.1795530390d+01,
     .    -0.2037561161d+00,0.1311572856d-01,-0.4436247106d-03,
     .    0.6107788810d-05,0.5225589273d-01,-0.9999933235d-02,    !.6
     .    -0.8995946611d+01,0.3504685691d+02,-0.5099531526d+02,
     .    0.1787058401d+02,-0.1697147581d+02,0.2029610875d+01,
     .    -0.2303135952d+00,0.1481207193d-01,-0.5002465600d-03,
     .    0.6874240214d-05,0.5152622406d-01,-0.9999960519d-02,    !.7
     .    0.5144665839d+00,-0.4844731614d+01,0.1363855009d+02,
     .    -0.1238845456d+02,-0.3787878816d+01,-0.8427313739d+00,
     .    0.1111571707d+00,-0.8085504025d-02,0.3012732411d-03,
     .    -0.4478276336d-05,0.1213491615d+01,-0.4371007555d-02,   !.8
     .    -0.1063931612d+02,0.4134755623d+02,-0.6041175583d+02,
     .    0.2112938887d+02,-0.2022753137d+02,0.2433883122d+01,
     .    -0.2777162211d+00,0.1795288826d-01,-0.6091176749d-03,
     .    0.8403997285d-05,0.5063316064d-01,-0.9893830933d-02,    !.9
     .    -0.1119324590d+02,0.4327356603d+02,-0.6297317075d+02,
     .    0.2122049821d+02,-0.2105758864d+02,0.2467390086d+01,
     .    -0.2797605528d+00,0.1799932187d-01,-0.6086973740d-03,
     .    0.8380938939d-05,0.5309274555d-01,-0.9999154558d-02/    !1.
C-----Reaction rate parameters
	DATA ((ratepar(i,j),j=1,13),i=1,11) /
C	ty ti tj tg th tk tl ni nj ng nh nk nl
C	--- -- -- -- -- -- -- -- -- -- -- -- --
     | 1, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 1,   !(1)!N->P
     | 1, 4, 0, 0, 0, 0, 5, 1, 0, 0, 0, 0, 1,   !(2)!H3->He3
     | 4,10, 0, 0, 0, 0, 6, 1, 0, 0, 0, 0, 2,   !(3)!Li8->2He4
     | 1,16, 0, 0, 0, 0,17, 1, 0, 0, 0, 0, 1,   !(4)!B12->C12
     | 1,21, 0, 0, 0, 0,22, 1, 0, 0, 0, 0, 1,   !(5)!C14->N14
     | 4,11, 0, 0, 0, 0, 6, 1, 0, 0, 0, 0, 2,   !(6)!B8->2He4
     | 1,15, 0, 0, 0, 0,14, 1, 0, 0, 0, 0, 1,   !(7)!C11->B11
     | 1,18, 0, 0, 0, 0,17, 1, 0, 0, 0, 0, 1,   !(8)!N12->C12
     | 1,20, 0, 0, 0, 0,19, 1, 0, 0, 0, 0, 1,   !(9)!N13->C13
     | 1,23, 0, 0, 0, 0,22, 1, 0, 0, 0, 0, 1,   !(10)!O14->N14
     | 1,25, 0, 0, 0, 0,24, 1, 0, 0, 0, 0, 1 /  !(11)!O15->N15
	DATA ((ratepar(i,j),j=1,13),i=12,22) /
C	ty ti tj tg th tk tl ni nj ng nh nk nl
C	--- -- -- -- -- -- -- -- -- -- -- -- --
     | 2, 2, 1, 0, 0, 0, 3, 1, 1, 0, 0, 0, 1,   !(12)!H(n,g)H2
     | 2, 3, 1, 0, 0, 0, 4, 1, 1, 0, 0, 0, 1,   !(13)!H2(n,g)H3
     | 2, 5, 1, 0, 0, 0, 6, 1, 1, 0, 0, 0, 1,   !(14)!He3(n,g)He4
     | 2, 7, 1, 0, 0, 0, 8, 1, 1, 0, 0, 0, 1,   !(15)!Li6(n,g)Li7
     | 3, 5, 1, 0, 0, 2, 4, 1, 1, 0, 0, 1, 1,   !(16)!He3(n,p)H3
     | 3, 9, 1, 0, 0, 2, 8, 1, 1, 0, 0, 1, 1,   !(17)!Be7(n,p)Li7
     | 3, 7, 1, 0, 0, 4, 6, 1, 1, 0, 0, 1, 1,   !(18)!Li6(n,t)He4
     | 5, 9, 1, 0, 0, 0, 6, 1, 1, 0, 0, 0, 2,   !(19)!Be7(n,a)He4
     | 2, 3, 2, 0, 0, 0, 5, 1, 1, 0, 0, 0, 1,   !(20)!H2(p,g)He3
     | 2, 4, 2, 0, 0, 0, 6, 1, 1, 0, 0, 0, 1,   !(21)!H3(p,g)He4
     | 2, 7, 2, 0, 0, 0, 9, 1, 1, 0, 0, 0, 1 /  !(22)!Li6(p,g)Be7
	DATA ((ratepar(i,j),j=1,13),i=23,33) /
C	ty ti tj tg th tk tl ni nj ng nh nk nl
C	--- -- -- -- -- -- -- -- -- -- -- -- --
     | 3, 7, 2, 0, 0, 5, 6, 1, 1, 0, 0, 1, 1,   !(23)!Li6(p,He3)He4
     | 5, 8, 2, 0, 0, 0, 6, 1, 1, 0, 0, 0, 2,   !(24)!Li7(p,a)He4
     | 2, 6, 3, 0, 0, 0, 7, 1, 1, 0, 0, 0, 1,   !(25)!He4(d,g)Li6
     | 2, 6, 4, 0, 0, 0, 8, 1, 1, 0, 0, 0, 1,   !(26)!He4(t,g)Li7
     | 2, 6, 5, 0, 0, 0, 9, 1, 1, 0, 0, 0, 1,   !(27)!He4(He3,g)Be7
     | 6, 3, 0, 0, 0, 1, 5, 2, 0, 0, 0, 1, 1,   !(28)!H2(d,n)He3
     | 6, 3, 0, 0, 0, 2, 4, 2, 0, 0, 0, 1, 1,   !(29)!H2(d,p)H3
     | 3, 4, 3, 0, 0, 1, 6, 1, 1, 0, 0, 1, 1,   !(30)!H3(d,n)He4
     | 3, 5, 3, 0, 0, 2, 6, 1, 1, 0, 0, 1, 1,   !(31)!He3(d,p)He4
     |11, 5, 0, 0, 0, 2, 6, 2, 0, 0, 0, 2, 1,   !(32)!He3(He3,2p)He4
     | 9, 8, 3, 0, 0, 1, 6, 1, 1, 0, 0, 1, 2 /  !(33)!Li7(d,na)He4
	DATA ((ratepar(i,j),j=1,13),i=34,44) /
C	ty ti tj tg th tk tl ni nj ng nh nk nl
C	--- -- -- -- -- -- -- -- -- -- -- -- --
     | 9, 9, 3, 0, 0, 2, 6, 1, 1, 0, 0, 1, 2,   !(34)!Be7(d,pa)He4
     | 2, 5, 4, 0, 0, 0, 7, 1, 1, 0, 0, 0, 1,   !(35)!He3(t,g)Li6
     | 3, 7, 3, 0, 0, 1, 9, 1, 1, 0, 0, 1, 1,   !(36)!Li6(d,n)Be7
     | 3, 7, 3, 0, 0, 2, 8, 1, 1, 0, 0, 1, 1,   !(37)!Li6(d,p)Li7
     | 3, 5, 4, 0, 0, 3, 6, 1, 1, 0, 0, 1, 1,   !(38)!He3(t,d)He4
     |11, 4, 0, 0, 0, 1, 6, 2, 0, 0, 0, 2, 1,   !(39)!H3(t,2n)He4
     |12, 5, 4, 0, 2, 1, 6, 1, 1, 0, 1, 1, 1,   !(40)!He3(t,np)He4
     | 3, 8, 4, 0, 0, 1,12, 1, 1, 0, 0, 1, 1,   !(41)!Li7(t,n)Be9
     | 3, 9, 4, 0, 0, 2,12, 1, 1, 0, 0, 1, 1,   !(42)!Be7(t,p)Be9
     | 3, 8, 5, 0, 0, 2,12, 1, 1, 0, 0, 1, 1,   !(43)!Li7(He3,p)Be9
     | 2, 8, 1, 0, 0, 0,10, 1, 1, 0, 0, 0, 1 /  !(44)!Li7(n,g)Li8
	DATA ((ratepar(i,j),j=1,13),i=45,55) /
C	ty ti tj tg th tk tl ni nj ng nh nk nl
C	--- -- -- -- -- -- -- -- -- -- -- -- --
     | 2,13, 1, 0, 0, 0,14, 1, 1, 0, 0, 0, 1,   !(45)!B10(n,g)B11
     | 2,14, 1, 0, 0, 0,16, 1, 1, 0, 0, 0, 1,   !(46)!B11(n,g)B12
     | 3,15, 1, 0, 0, 2,14, 1, 1, 0, 0, 1, 1,   !(47)!C11(n,p)B11
     | 3,13, 1, 0, 0, 6, 8, 1, 1, 0, 0, 1, 1,   !(48)!B10(n,a)Li7
     | 2, 9, 2, 0, 0, 0,11, 1, 1, 0, 0, 0, 1,   !(49)!Be7(p,g)B8
     | 2,12, 2, 0, 0, 0,13, 1, 1, 0, 0, 0, 1,   !(50)!Be9(p,g)B10
     | 2,13, 2, 0, 0, 0,15, 1, 1, 0, 0, 0, 1,   !(51)!B10(p,g)C11
     | 2,14, 2, 0, 0, 0,17, 1, 1, 0, 0, 0, 1,   !(52)!B11(p,g)C12
     | 2,15, 2, 0, 0, 0,18, 1, 1, 0, 0, 0, 1,   !(53)!C11(p,g)N12
     | 3,16, 2, 0, 0, 1,17, 1, 1, 0, 0, 1, 1,   !(54)!B12(p,n)C12
     | 3,12, 2, 0, 0, 6, 7, 1, 1, 0, 0, 1, 1 /  !(55)!Be9(p,a)Li6
	DATA ((ratepar(i,j),j=1,13),i=56,66) /
C	ty ti tj tg th tk tl ni nj ng nh nk nl
C	--- -- -- -- -- -- -- -- -- -- -- -- --
     | 3,13, 2, 0, 0, 6, 9, 1, 1, 0, 0, 1, 1,   !(56)!B10(p,a)Be7
     | 3,16, 2, 0, 0, 6,12, 1, 1, 0, 0, 1, 1,   !(57)!B12(p,a)Be9
     | 2, 7, 6, 0, 0, 0,13, 1, 1, 0, 0, 0, 1,   !(58)!Li6(a,g)B10
     | 2, 8, 6, 0, 0, 0,14, 1, 1, 0, 0, 0, 1,   !(59)!Li7(a,g)B11
     | 2, 9, 6, 0, 0, 0,15, 1, 1, 0, 0, 0, 1,   !(60)!Be7(a,g)C11
     | 3,11, 6, 0, 0, 2,15, 1, 1, 0, 0, 1, 1,   !(61)!B8(a,p)C11
     | 3,10, 6, 0, 0, 1,14, 1, 1, 0, 0, 1, 1,   !(62)!Li8(a,n)B11
     | 3,12, 6, 0, 0, 1,17, 1, 1, 0, 0, 1, 1,   !(63)!Be9(a,n)C12
     | 3,12, 3, 0, 0, 1,13, 1, 1, 0, 0, 1, 1,   !(64)!Be9(d,n)B10
     | 3,13, 3, 0, 0, 2,14, 1, 1, 0, 0, 1, 1,   !(65)!B10(d,p)B11
     | 3,14, 3, 0, 0, 1,17, 1, 1, 0, 0, 1, 1 /  !(66)!B11(d,n)C12
	DATA ((ratepar(i,j),j=1,13),i=67,77) /
C	ty ti tj tg th tk tl ni nj ng nh nk nl
C	--- -- -- -- -- -- -- -- -- -- -- -- --
     | 8, 6, 1, 0, 0, 0,12, 2, 1, 0, 0, 0, 1,   !(67)!He4(an,g)Be9
     | 7, 6, 0, 0, 0, 0,17, 3, 0, 0, 0, 0, 1,   !(68)!He4(2a,g)C12
     | 9,10, 2, 0, 0, 1, 6, 1, 1, 0, 0, 1, 2,   !(69)!Li8(p,na)He4
     | 9,11, 1, 0, 0, 2, 6, 1, 1, 0, 0, 1, 2,   !(70)!B8(n,pa)He4
     | 9,12, 2, 0, 0, 3, 6, 1, 1, 0, 0, 1, 2,   !(71)!Be9(p,da)He4
     |10,14, 2, 0, 0, 0, 6, 1, 1, 0, 0, 0, 3,   !(72)!B11(p,2a)He4
     |10,15, 1, 0, 0, 0, 6, 1, 1, 0, 0, 0, 3,   !(73)!C11(n,2a)He4
     | 2,17, 1, 0, 0, 0,19, 1, 1, 0, 0, 0, 1,   !(74)!C12(n,g)C13
     | 2,19, 1, 0, 0, 0,21, 1, 1, 0, 0, 0, 1,   !(75)!C13(n,g)C14
     | 2,22, 1, 0, 0, 0,24, 1, 1, 0, 0, 0, 1,   !(76)!N14(n,g)N15
     | 3,20, 1, 0, 0, 2,19, 1, 1, 0, 0, 1, 1 /  !(77)!N13(n,p)C13
	DATA ((ratepar(i,j),j=1,13),i=78,88) /
C	ty ti tj tg th tk tl ni nj ng nh nk nl
C	--- -- -- -- -- -- -- -- -- -- -- -- --
     | 3,22, 1, 0, 0, 2,21, 1, 1, 0, 0, 1, 1,   !(78)!N14(n,p)C14
     | 3,25, 1, 0, 0, 2,24, 1, 1, 0, 0, 1, 1,   !(79)!O15(n,p)N15
     | 3,25, 1, 0, 0, 6,17, 1, 1, 0, 0, 1, 1,   !(80)!O15(n,a)C12
     | 2,17, 2, 0, 0, 0,20, 1, 1, 0, 0, 0, 1,   !(81)!C12(p,g)N13
     | 2,19, 2, 0, 0, 0,22, 1, 1, 0, 0, 0, 1,   !(82)!C13(p,g)N14
     | 2,21, 2, 0, 0, 0,24, 1, 1, 0, 0, 0, 1,   !(83)!C14(p,g)N15
     | 2,20, 2, 0, 0, 0,23, 1, 1, 0, 0, 0, 1,   !(84)!N13(p,g)O14
     | 2,22, 2, 0, 0, 0,25, 1, 1, 0, 0, 0, 1,   !(85)!N14(p,g)O15
     | 2,24, 2, 0, 0, 0,26, 1, 1, 0, 0, 0, 1,   !(86)!N15(p,g)O16
     | 3,24, 2, 0, 0, 6,17, 1, 1, 0, 0, 1, 1,   !(87)!N15(p,a)C12
     | 2,17, 6, 0, 0, 0,26, 1, 1, 0, 0, 0, 1 /  !(88)!C12(a,g)O16
	DATA ((ratepar(i,j),j=1,13),i=89,99) /
C	ty ti tj tg th tk tl ni nj ng nh nk nl
C	--- -- -- -- -- -- -- -- -- -- -- -- --
     | 3,13, 6, 0, 0, 2,19, 1, 1, 0, 0, 1, 1,   !(89)!B10(a,p)C13
     | 3,14, 6, 0, 0, 2,21, 1, 1, 0, 0, 1, 1,   !(90)!B11(a,p)C14
     | 3,15, 6, 0, 0, 2,22, 1, 1, 0, 0, 1, 1,   !(91)!C11(a,p)N14
     | 3,18, 6, 0, 0, 2,25, 1, 1, 0, 0, 1, 1,   !(92)!N12(a,p)O15
     | 3,20, 6, 0, 0, 2,26, 1, 1, 0, 0, 1, 1,   !(93)!N13(a,p)O16
     | 3,13, 6, 0, 0, 1,20, 1, 1, 0, 0, 1, 1,   !(94)!B10(a,n)N13
     | 3,14, 6, 0, 0, 1,22, 1, 1, 0, 0, 1, 1,   !(95)!B11(a,n)N14
     | 3,16, 6, 0, 0, 1,24, 1, 1, 0, 0, 1, 1,   !(96)!B12(a,n)N15
     | 3,19, 6, 0, 0, 1,26, 1, 1, 0, 0, 1, 1,   !(97)!C13(a,n)O16
     | 3,14, 3, 0, 0, 2,16, 1, 1, 0, 0, 1, 1,   !(98)!B11(d,p)B12
     | 3,17, 3, 0, 0, 2,19, 1, 1, 0, 0, 1, 1 /  !(99)!C12(d,p)C13
	DATA ((ratepar(i,j),j=1,13),i=100,100) /
C	ty ti tj tg th tk tl ni nj ng nh nk nl
C	--- -- -- -- -- -- -- -- -- -- -- -- --
     | 3,19, 3, 0, 0, 2,21, 1, 1, 0, 0, 1, 1 /  !(100)!C13(d,p)C14
C-----Text strings for the reactions
	DATA rstring/
C	---i=1,11
     . 'N->P',
     . 'H3->He3',
     . 'Li8->2He4',
     . 'B12->C12',
     . 'C14->N14',
     . 'B8->2He4',
     . 'C11->B11',
     . 'N12->C12',
     . 'N13->C13',
     . 'O14->N14',
     . 'O15->N15',
C	---i=12,22
     . 'H(n,g)H2',
     . 'H2(n,g)H3',
     . 'He3(n,g)He4',
     . 'Li6(n,g)Li7',
     . 'He3(n,p)H3',
     . 'Be7(n,p)Li7',
     . 'Li6(n,t)He4',
     . 'Be7(n,a)He4',
     . 'H2(p,g)He3',
     . 'H3(p,g)He4',
     . 'Li6(p,g)Be7',
C	---i=23,33
     . 'Li6(p,He3)He4',
     . 'Li7(p,a)He4',
     . 'He4(d,g)Li6',
     . 'He4(t,g)Li7',
     . 'He4(He3,g)Be7',
     . 'H2(d,n)He3',
     . 'H2(d,p)H3',
     . 'H3(d,n)He4',
     . 'He3(d,p)He4',
     . 'He3(He3,2p)He4',
     . 'Li7(d,na)He4',
C	---i=34,44
     . 'Be7(d,pa)He4',
     . 'He3(t,g)Li6',
     . 'Li6(d,n)Be7',
     . 'Li6(d,p)Li7',
     . 'He3(t,d)He4',
     . 'H3(t,2n)He4',
     . 'He3(t,np)He4',
     . 'Li7(t,n)Be9',
     . 'Be7(t,p)Be9',
     . 'Li7(He3,p)Be9',
     . 'Li7(n,g)Li8',
C	---i=45,55
     . 'B10(n,g)B11',
     . 'B11(n,g)B12',
     . 'C11(n,p)B11',
     . 'B10(n,a)Li7',
     . 'Be7(p,g)B8',
     . 'Be9(p,g)B10',
     . 'B10(p,g)C11',
     . 'B11(p,g)C12',
     . 'C11(p,g)N12',
     . 'B12(p,n)C12',
     . 'Be9(p,a)Li6',
C	---i=56,66
     . 'B10(p,a)Be7',
     . 'B12(p,a)Be9',
     . 'Li6(a,g)B10',
     . 'Li7(a,g)B11',
     . 'Be7(a,g)C11',
     . 'B8(a,p)C11',
     . 'Li8(a,n)B11',
     . 'Be9(a,n)C12',
     . 'Be9(d,n)B10',
     . 'B10(d,p)B11',
     . 'B11(d,n)C12',
C	---i=67,77
     . 'He4(an,g)Be9',
     . 'He4(2a,g)C12',
     . 'Li8(p,na)He4',
     . 'B8(n,pa)He4',
     . 'Be9(p,da)He4',
     . 'B11(p,2a)He4',
     . 'C11(n,2a)He4',
     . 'C12(n,g)C13',
     . 'C13(n,g)C14',
     . 'N14(n,g)N15',
     . 'N13(n,p)C13',
C	---i=78,88
     . 'N14(n,p)C14',
     . 'O15(n,p)N15',
     . 'O15(n,a)C12',
     . 'C12(p,g)N13',
     . 'C13(p,g)N14',
     . 'C14(p,g)N15',
     . 'N13(p,g)O14',
     . 'N14(p,g)O15',
     . 'N15(p,g)O16',
     . 'N15(p,a)C12',
     . 'C12(a,g)O16',
C	---i=89,99
     . 'B10(a,p)C13',
     . 'B11(a,p)C14',
     . 'C11(a,p)N14',
     . 'N12(a,p)O15',
     . 'N13(a,p)O16',
     . 'B10(a,n)N13',
     . 'B11(a,n)N14',
     . 'B12(a,n)N15',
     . 'C13(a,n)O16',
     . 'B11(d,p)B12',
     . 'C12(d,p)C13',
C	---i=100,100
     . 'C13(d,p)C14'/

	end

      SUBROUTINE ZBRAK(FX,X1,X2,N,XB1,XB2,NB)
C     Based on the routine of the same name in Numerical Recipes.
C     Given a function fx defined on the interval from x1-x2 subdivide the interval into n equally
C     spaced segments, and search for zero crossings of the function. nb is input as the maximum
C     number of roots sought, and is reset to the number of bracketing pairs xb1(1:nb),
C     xb2(1:nb) that are found.
C
C     Called by INIT
C     Calls FX(=LHFUN)
C
      IMPLICIT NONE
      INTEGER N,NB
      REAL*8 X1,X2,XB1(NB),XB2(NB),FX
      EXTERNAL FX
      INTEGER I,NBB
      REAL*8 DX,FC,FP,X

      nbb=0
      x=x1
      dx=(x2-x1)/n
      fp=fx(x)
      do i=1,n
        x=x+dx
        fc=fx(x)
        if(fc*fp.le.0.d0) then
          nbb=nbb+1
          xb1(nbb)=x-dx
          xb2(nbb)=x
          if(nbb.eq.nb)goto 1
        endif
        fp=fc
      enddo
1     continue
      nb=nbb

      RETURN
      END


      FUNCTION ZBRENT(FUNC,X1,X2,TOL)
C     Based on the routine of the same name in Numerical Recipes.
C     Using Brent's method, find the root of a function func known to lie between x1 and x2.
C     The root, returned as zbrent, will be rened until its accuracy is tol.
C     Parameters: Maximum allowed number of iterations, and machine floating-point precision.
C
C     Called by INIT
C     Calls FUNC(=LHFUN)
C
      IMPLICIT NONE
      INTEGER ITMAX
      REAL*8 ZBRENT,TOL,X1,X2,FUNC,EPS
      EXTERNAL FUNC
      PARAMETER (ITMAX=100,EPS=3.d-8)
      INTEGER ITER
      REAL*8 A,B,C,D,E,FA,FB,FC,P,Q,R,S,TOL1,XM

      a=x1
      b=x2
      fa=func(a)
      fb=func(b)
      if((fa.gt.0.d0.and.fb.gt.0.d0).or.(fa.lt.0.d0.and.fb.lt.0.d0))
     .then
        write(*,*) ' Error in zbrent: root must be bracketed'
        stop
      endif
      c=b
      fc=fb
      do iter=1,ITMAX
        if((fb.gt.0.d0.and.fc.gt.0.d0).or.(fb.lt.0.d0.and.fc.lt.0.d0))
     .then
          c=a
          fc=fa
          d=b-a
          e=d
        endif
        if(abs(fc).lt.abs(fb)) then
          a=b
          b=c
          c=a
          fa=fb
          fb=fc
          fc=fa
        endif
        tol1=2.d0*EPS*abs(b)+0.5d0*tol
        xm=.5d0*(c-b)
        if(abs(xm).le.tol1 .or. fb.eq.0.d0)then
          zbrent=b
          return
        endif
        if(abs(e).ge.tol1 .and. abs(fa).gt.abs(fb)) then
          s=fb/fa
          if(a.eq.c) then
            p=2.d0*xm*s
            q=1.d0-s
          else
            q=fa/fc
            r=fb/fc
            p=s*(2.d0*xm*q*(q-r)-(b-a)*(r-1.d0))
            q=(q-1.d0)*(r-1.d0)*(s-1.d0)
          endif
          if(p.gt.0.d0) q=-q
          p=abs(p)
          if(2.d0*p .lt. min(3.d0*xm*q-abs(tol1*q),abs(e*q))) then
            e=d
            d=p/q
          else
            d=xm
            e=d
          endif
        else
          d=xm
          e=d
        endif
        a=b
        fa=fb
        if(abs(d) .gt. tol1) then
          b=b+d
        else
          b=b+sign(tol1,xm)
        endif
        fb=func(b)
      enddo
      write(*,*) ' zbrent exceeding maximum iterations'
      zbrent=b

      RETURN
      END


      FUNCTION BESSI0(X)
C     Based on the routine of the same name in Numerical Recipes.
C     Returns the modified Bessel function I0(x) for any real x.
C
C     Called by BESSK0
C
      IMPLICIT NONE
      REAL*8 BESSI0,X
      REAL*8 AX
      DOUBLE PRECISION P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Y
      SAVE P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9
      DATA P1,P2,P3,P4,P5,P6,P7/1.0D0,3.5156229D0,3.0899424D0,
     *1.2067492D0,0.2659732D0,0.360768D-1,0.45813D-2/
      DATA Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9/0.39894228D0,0.1328592D-1,
     *0.225319D-2,-0.157565D-2,0.916281D-2,-0.2057706D-1,0.2635537D-1,
     *-0.1647633D-1,0.392377D-2/

      if (abs(x).lt.3.75) then
        y=(x/3.75)**2
        bessi0=p1+y*(p2+y*(p3+y*(p4+y*(p5+y*(p6+y*p7)))))
      else
        ax=abs(x)
        y=3.75/ax
        bessi0=(exp(ax)/sqrt(ax))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*
     *(q7+y*(q8+y*q9))))))))
      endif

      RETURN
      END


      FUNCTION BESSK0(X)
C     Based on the routine of the same name in Numerical Recipes.
C     Returns the modified Bessel function K0(x) for positive real x.
C
C     Called by BESSK,BESSEL
C     Calls BESSI0
C
      IMPLICIT NONE
      REAL*8 BESSK0,X
      REAL*8 BESSI0
      DOUBLE PRECISION P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Y
      SAVE P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7
      DATA P1,P2,P3,P4,P5,P6,P7/-0.57721566D0,0.42278420D0,0.23069756D0,
     *0.3488590D-1,0.262698D-2,0.10750D-3,0.74D-5/
      DATA Q1,Q2,Q3,Q4,Q5,Q6,Q7/1.25331414D0,-0.7832358D-1,0.2189568D-1,
     *-0.1062446D-1,0.587872D-2,-0.251540D-2,0.53208D-3/

      if (x.le.2.0) then
        y=x*x/4.0
        bessk0=(-log(x/2.0)*bessi0(x))+(p1+y*(p2+y*(p3+y*(p4+y*(p5+y*
     *(p6+y*p7))))))
      else
        y=(2.0/x)
        bessk0=(exp(-x)/sqrt(x))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*
     *q7))))))
      endif

      RETURN
      END


      FUNCTION BESSI1(X)
C     Based on the routine of the same name in Numerical Recipes.
C     Returns the modified Bessel function I1(x) for any real x.
C
C     Called by BESSK1
C
      IMPLICIT NONE
      REAL*8 BESSI1,X
      REAL*8 AX
      DOUBLE PRECISION P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Y
      SAVE P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9
      DATA P1,P2,P3,P4,P5,P6,P7/0.5D0,0.87890594D0,0.51498869D0,
     *0.15084934D0,0.2658733D-1,0.301532D-2,0.32411D-3/
      DATA Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9/0.39894228D0,-0.3988024D-1,
     *-0.362018D-2,0.163801D-2,-0.1031555D-1,0.2282967D-1,-0.2895312D-1,
     *0.1787654D-1,-0.420059D-2/

      if (abs(x).lt.3.75) then
        y=(x/3.75)**2
        bessi1=x*(p1+y*(p2+y*(p3+y*(p4+y*(p5+y*(p6+y*p7))))))
      else
        ax=abs(x)
        y=3.75/ax
        bessi1=(exp(ax)/sqrt(ax))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*
     *(q7+y*(q8+y*q9))))))))
        if(x.lt.0.)bessi1=-bessi1
      endif

      RETURN
      END


      FUNCTION BESSK1(X)
C     Based on the routine of the same name in Numerical Recipes.
C     Returns the modified Bessel function K1(x) for positive real x.
C
C     Called by BESSK,BESSEL
C     Calls BESSI1
C
      IMPLICIT NONE
      REAL*8 BESSK1,X
      REAL*8 BESSI1
      DOUBLE PRECISION P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Y
      SAVE P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7
      DATA P1,P2,P3,P4,P5,P6,P7/1.0D0,0.15443144D0,-0.67278579D0,
     *-0.18156897D0,-0.1919402D-1,-0.110404D-2,-0.4686D-4/
      DATA Q1,Q2,Q3,Q4,Q5,Q6,Q7/1.25331414D0,0.23498619D0,-0.3655620D-1,
     *0.1504268D-1,-0.780353D-2,0.325614D-2,-0.68245D-3/

      if (x.le.2.0) then
        y=x*x/4.0
        bessk1=(log(x/2.0)*bessi1(x))+(1.0/x)*(p1+y*(p2+y*(p3+y*(p4+y*
     *(p5+y*(p6+y*p7))))))
      else
        y=2.0/x
        bessk1=(exp(-x)/sqrt(x))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*
     *q7))))))
      endif

      RETURN
      END


      FUNCTION BESSK(N,X)
C     Based on the routine of the same name in Numerical Recipes.
C     Returns the modified Bessel function Kn(x) for positive x and n>=2.
C
C     Called by BESSEL
C     Calls BESSK0,BESSK1
C
      IMPLICIT NONE
      INTEGER N
      REAL*8 BESSK,X
      INTEGER J
      REAL*8 BK,BKM,BKP,TOX,BESSK0,BESSK1

      if (n.lt.2) then
        write(*,*) ' bad argument n in bessk'
        stop
      endif
      tox=2.0/x
      bkm=bessk0(x)
      bk=bessk1(x)
      do 11 j=1,n-1
        bkp=bkm+j*tox*bk
        bkm=bk
        bk=bkp
11    continue
      bessk=bk

      RETURN
      END


      FUNCTION BESSEL(N,X)
C
C     Called by THERMO
C     Calls BESSK0,BESSK1,BESSK
C
      IMPLICIT NONE
      INTEGER N
      REAL*8 BESSEL,BESSK0,BESSK1,BESSK,X

      if (n.lt.0) then
        write(*,*) ' Bad argument in bessel: n must be not negative'
        stop
      elseif (n.eq.0) then
        bessel=bessk0(x)
      elseif (n.eq.1) then
        bessel=bessk1(x)
      elseif (n.ge.2) then
        bessel=bessk(n,x)
      endif

      RETURN
      END


      SUBROUTINE TRAPZD(FUNC,A,B,S,N)
C     Based on the routine of the same name in Numerical Recipes.
C     This routine computes the nth stage of refinement of an extended trapezoidal rule. func is
C     input as the name of the function to be integrated between limits a and b, also input. When
C     called with n=1, the routine returns as s the crudest estimate of integral_a^b f(x)dx. Subsequent
C     calls with n=2,3,... (in that sequential order) will improve the accuracy of s by adding 2^(n-2)
C     additional interior points. s should not be modied between sequential calls.
C
C     Called by QTRAP
C
      IMPLICIT DOUBLE PRECISION (A-Z)
      INTEGER N
      EXTERNAL FUNC
      INTEGER IT,J

      if (n.eq.1) then
        s=0.5d0*(b-a)*(func(a)+func(b))
      else
        it=2**(n-2)
        tnm=it
        del=(b-a)/tnm
        x=a+0.5d0*del
        sum=0.0d0
        do j=1,it
          sum=sum+func(x)
          x=x+del
        enddo
        s=0.5d0*(s+(b-a)*sum/tnm)
      endif

      RETURN
      END


      SUBROUTINE QTRAP(FUNC,A,B,S,EPS,IER)
C     Based on the routine of the same name in Numerical Recipes.
C     Returns as s the integral of the function func from a to b. The parameters EPS can be set
C     to the desired fractional accuracy and JMAX so that 2^(JMAX-1) is the maximum
C     allowed number of steps. Integration is performed by the trapezoidal rule.
C
C     Called by LHFUN
C     Calls TRAPZD
C
      IMPLICIT DOUBLE PRECISION (A-Z)
      INTEGER IER,JMAX
      EXTERNAL FUNC
      PARAMETER (JMAX=20)
      INTEGER J

      olds=-1.d30
      do j=1,jmax
        call trapzd(func,a,b,s,j)
        if (abs(s-olds).lt.eps*abs(olds)) return
        olds=s
      enddo
      ier=1

      RETURN
      END

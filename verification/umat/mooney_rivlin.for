C     Compressible Mooney-Rivlin UHYPER, U = c1/2 (I1bar - 3) + c2/2 (I2bar - 3) + 1/D1 (J - 1)^2,
C     PROPS = (D1, c1, c2)
      SUBROUTINE UHYPER(BI1,BI2,AJ,U,UI1,UI2,UI3,TEMP,NOEL,CMNAME,
     1 INCMPFLAG,NUMSTATEV,STATEV,NUMFIELDV,FIELDV,FIELDVINC,
     2 NUMPROPS,PROPS)
      INCLUDE 'ABA_PARAM.INC'
      CHARACTER*80 CMNAME
      DIMENSION UI1(3),UI2(6),UI3(6),STATEV(*),FIELDV(*),
     1 FIELDVINC(*),PROPS(*)
      D1=PROPS(1)
      C1=PROPS(2)
      C2=PROPS(3)
      U=C1/2.D0*(BI1-3.D0)+C2/2.D0*(BI2-3.D0)+(AJ-1.D0)**2/D1
      UI1(1)=C1/2.D0
      UI1(2)=C2/2.D0
      UI1(3)=2.D0/D1*(AJ-1.D0)
      DO K=1,6
        UI2(K)=0.D0
        UI3(K)=0.D0
      END DO
      UI2(3)=2.D0/D1
      RETURN
      END

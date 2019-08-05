      function multReal(a1, a2, b1, b2) result(c)
      implicit none
      real*8 a1, a2, b1, b2, c
      c = a1*b1 - a2*b2
      return 
      end function

      function multImag(a1, a2, b1, b2) result(c)
      implicit none
      real*8 a1, a2, b1, b2, c
      c = a1*b2 + a2*b1
      return
      end function

      subroutine multCmplx(a1, a2, b1, b2, c)
      implicit none
      real*8 a1, a2, b1, b2, c(2), multReal, multImag
      c(1) = multReal(a1,a2,b1,b2)
      c(2) = multImag(a1,a2,b1,b2)
      end subroutine multCmplx

      !ind for a lower triangular matrix
      function indUpper(i, j) result(indOut)
      implicit none
      integer i, j, indOut

      indOut = max(i,j)*(max(i,j)+1)/2 + min(i,j) + 1
      return
      end function


      !c-style ind for a full matrix
      function ind(i,j, ncols) result(indOut)
      implicit none
      integer i, j, ncols, indOut

      indOut = i*ncols + j + 1
      return
      end function

c************************************************
c evalbraJ
c************************************************
      subroutine evalbraJ(braJ, bra, JAi, braNcol, nelec, norbs)
      implicit none

      integer braNcol, nelec, i, j, norbs, ind
      real*8  braJ(2*norbs*nelec)
      real*8 bra(2*norbs*nelec)
      real*8  JAi(norbs)

      !braJ = exp(J) * bra
      do i = 1,norbs
         do j = 1,nelec
            braJ(2*ind(i-1, j-1, nelec)-1)
     c           = bra(2*ind(i-1, j-1, braNcol)-1) 
     c           * exp(-2.*JAi(i))

            braJ(2*ind(i-1, j-1, nelec))
     c           = bra(2*ind(i-1, j-1, braNcol)) 
     c           * exp(-2.*JAi(i))
         end do
      end do
      end subroutine evalbraJ

c************************************************
c evalS
c************************************************
      subroutine evalS(braJ, ketJ, S, norbs,
     c     ketJNcol, nelec)

      implicit none
      integer norbs, nelec, i, j, k, ind
      integer braNcol, ketJNcol, ind1, ind2, ind3
 
      real*8  S  (2*nelec*nelec), dummy(2)
      real*8  ketJ(2*norbs*nelec)
      real*8  braJ(2*norbs*nelec)

      do i = 1,2*nelec*nelec
         S(i) = 0.0
      end do

      !S = braJ^T ketJ
      !S-> nelec, nelec
      !braJ -> norbs, nelec
      !ketJ -> norbs, nelec
      do i = 1,nelec
         do k = 1,norbs
            do j = 1,nelec
               ind1 = ind(i-1, j-1, nelec)
               ind2 = ind(k-1, i-1, nelec)
               ind3 = ind(k-1, j-1, nelec)

               call multCmplx(braJ(2*ind2 -1), braJ(2*ind2), 
     c              ketJ(2*ind3-1), ketJ(2*ind3), dummy)
               S(2*ind1-1) = S(2*ind1-1) + dummy(1)
               S(2*ind1  ) = S(2*ind1  ) + dummy(2)

!               S(ind(i-1, j-1, nelec)) = S(ind(i-1, j-1, nelec)) + 
!     c              braJ(ind(k-1, i-1, nelec)) 
!     c              * ketJ(ind(k-1, j-1, ketJNcol))
            end do
         end do
      end do
      end subroutine evalS



c************************************************
c evalRDM
c************************************************
      subroutine evalGrad1RDM(bra, JAi, Sinv, ketJ, braJGiven, norbs, 
     c     braNcol, SinvNcol, ketJNcol, braJGivenNcol,
     c     nelec, rdm,
     c     orb1, orb2, detS)
      implicit none

      integer norbs, nelec, orb1, orb2, i, j, ind, indUpper
      integer braNcol, SinvNcol, ketJNcol, ind1, ind2, ind3
      integer braJGivenNcol

      real*8  bra(2*norbs*nelec), detS(2), multReal
      real*8      JAi(norbs), rdm
      real*8  Sinv(2*nelec*nelec), braJGiven(2*norbs*nelec)
      real*8  ketJ(2*norbs*nelec)

      real*8   TrSinvS(2), rdmcmplx(2), dummy1(2), dummy2(2)
      real*8  S   (2*nelec*nelec)
      real*8  braJ(2*norbs*nelec)
      real*8  intermediate(2*nelec)
      real*8  intermediate2(2*nelec)

      call evalbraJ(braJ, bra, JAi, braNcol, nelec, norbs)

      !intermediate = ketJ * Sinv
      do j = 1,nelec
         intermediate(2*j-1) = 0.0
         intermediate(2*j)   = 0.0
         do i = 1,nelec
            ind1 = ind(orb2,   i-1, ketJNcol)
            ind2 = ind(i-1,   j-1, SinvNcol)
            call multCmplx(ketJ(2*ind1-1), ketJ(2*ind1),
     +           Sinv(2*ind2 -1), Sinv(2*ind2), dummy1)

            intermediate(2*j-1) = intermediate(2*j-1) + dummy1(1)
            intermediate(2*j  ) = intermediate(2*j  ) + dummy1(2)
         end do
      end do


      !*****THIS IS THE FIRST TERM
      !rdm = det(S) ketJ(orb2, i) * Sinv(i, j) * braJ(orb1, j)
      rdm = 0.0
      do j = 1,nelec
         call multCmplx(detS(1), detS(2), intermediate(2*j-1), 
     c        intermediate(2*j), dummy1)
         rdm = rdm
     c        + multReal(dummy1(1), dummy1(2),  
     c        braJ(2*ind(orb1, j-1, nelec)-1), 
     c        braJ(2*ind(orb1, j-1, nelec)))
      end do
      !***************************
      write(*,*) rdm


      call evalS(braJ, ketJ, S, norbs, ketJNcol, nelec)

      !*****THIS IS THE SECOND TERM
      !rdm = det(S) Tr(Sinv * S) ketJ(orb2, i) * Sinv(i, j) * braJGiven(orb1, j)
      TrSinvS(1) = 0. 
      TrSinvS(2) = 0.
      do i = 1, nelec
         do j = 1, nelec
            ind1 = 2*ind(i-1, j-1, SinvNcol) - 1
            ind2 = 2*ind(j-1, i-1, nelec) - 1
            call multCmplx(Sinv(ind1), Sinv(ind1+1), S(ind2), S(ind2+1), 
     c           dummy1)
            TrSinvS(1) = TrSinvS(1) + dummy1(1)
            TrSinvS(2) = TrSinvS(2) + dummy1(2)
         end do
      end do
      call multCmplx(detS(1), detS(2), TrSinvS(1), TrSinvS(2), dummy2)
      do j = 1,nelec
         ind1 = 2* ind(orb1, j-1, braJGivenNcol) - 1
         call multCmplx(intermediate(2*j-1), intermediate(2*j), 
     c        braJGiven(ind1), braJGiven(ind1+1), dummy1)

         rdm = rdm + multReal(dummy2(1), dummy2(2), dummy1(1),dummy1(2))
      end do
      !****************************
      write(*,*) rdm

      !intermediate2 = intermediate * S
      do i = 1,nelec
         intermediate2(2*i-1) = 0.0
         intermediate2(2*i)   = 0.0
!         intermediate2(i) = (0., 0.)
         do j = 1,nelec
            call multCmplx(intermediate(2*j-1), intermediate(2*j),
     c           S(2*ind(i-1,    j-1, SinvNcol)-1) ,
     c           S(2*ind(i-1,    j-1, SinvNcol)) , dummy1)

            intermediate2(2*i-1) = intermediate2(2*i-1) + dummy1(1)
            intermediate2(2*i  ) = intermediate2(2*i  ) + dummy1(2)
            
!            intermediate2(i) = intermediate2(i) +   
!     c           intermediate(j) * 
!     c           S(ind(i-1,    j-1, SinvNcol)) 
         end do
      end do

      !intermediate = intermediate2 * Sinv
      do i = 1,nelec
         intermediate(2*i-1) = 0.0
         intermediate(2*i)   = 0.0
!         intermediate(i) = (0., 0.)
         do j = 1,nelec
            call multCmplx(intermediate2(2*j-1), intermediate2(2*j),  
     c           Sinv(2*ind(i-1,    j-1, SinvNcol)-1), 
     c           Sinv(2*ind(i-1,    j-1, SinvNcol)) , dummy1)

            intermediate(2*i-1) = intermediate(2*i-1) + dummy1(1)
            intermediate(2*i  ) = intermediate(2*i  ) + dummy1(2)

!            intermediate(i) = intermediate(i) +   
!     c           intermediate2(j) * 
!     c           Sinv(ind(i-1,    j-1, SinvNcol)) 
         end do
      end do

      !*****THIS IS THE THIRD TERM
      !rdmcmplx += detS*intermediate*braJ
      do j = 1,nelec
         call multCmplx(detS(1), detS(2), intermediate(2*j-1), 
     c        intermediate(2*j), dummy1)
         rdm = rdm - multReal(dummy1(1), dummy1(2),
     c           braJGiven(2*ind(orb1, j-1, nelec)-1),
     c           braJGiven(2*ind(orb1, j-1, nelec)))
!         rdmcmplx = rdmcmplx - intermediate(j) * 
!     c           braJ(ind(orb1, j-1, nelec))
      end do

!      rdm = real(rdmcmplx) 
      write(*,*) rdm

      end subroutine evalGrad1RDM

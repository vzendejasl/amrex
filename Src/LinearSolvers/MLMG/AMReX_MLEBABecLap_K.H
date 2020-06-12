#ifndef AMREX_MLEBABECLAP_K_H_
#define AMREX_MLEBABECLAP_K_H_

#include <AMReX_MLLinOp_K.H>

#include <AMReX_EBCellFlag.H>

namespace amrex { namespace {
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real get_dx_eb (Real kappa) noexcept {
        return amrex::max(0.3_rt,(kappa*kappa-0.25_rt)/(2.0_rt*kappa));
    }
}}

namespace amrex {

// note that the mask in these functions is different from masks in bndry registers
// 1 means valid data, 0 means invalid data

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabeclap_apply_bc_x (int side, Box const& box, int blen,
                             Array4<Real> const& phi,
                             Array4<int const> const& mask,
                             Array4<Real const> const& area,
                             BoundCond bct, Real bcl,
                             Array4<Real const> const& bcval,
                             int maxorder, Real dxinv, int inhomog, int icomp) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);
    const int i = lo.x; // boundary cell
    const int s = 1-2*side;  // +1 for lo and -1 for hi
    switch (bct) {
    case AMREX_LO_NEUMANN:
    {
        for     (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                if (mask(i,j,k) == 0 and mask(i+s,j,k) == 1) {
                    phi(i,j,k,icomp) = phi(i+s,j,k,icomp);
                }
            }
        }
        break;
    }
    case AMREX_LO_REFLECT_ODD:
    {
        for     (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                if (mask(i,j,k) == 0 and mask(i+s,j,k) == 1) {
                    phi(i,j,k,icomp) = -phi(i+s,j,k,icomp);
                }
            }
        }
        break;
    }
    case AMREX_LO_DIRICHLET:
    {
        const int NX = amrex::min(blen+1, maxorder);
        GpuArray<Real,4> x{-bcl * dxinv, 0.5_rt, 1.5_rt, 2.5_rt};
        Array2D<Real, 0, 3, 0, 2> coef{};
        for (int r = 0; r <= maxorder-2; ++r) {
            poly_interp_coeff(-0.5_rt, &x[0], NX, &(coef(0,r)));
        }
        for     (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                if (mask(i,j,k) == 0 and mask(i+s,j,k) == 1) {
                    int order = 1;
                    for (int r = 0; r <= NX-2; ++r) {
                        if (area(i+(1-side)+s*r,j,k) > 0.0_rt) {
                            ++order;
                        } else {
                            break;
                        }
                    }
                    if (order == 1) {
                        if (inhomog) {
                            phi(i,j,k,icomp) = bcval(i,j,k,icomp);
                        } else {
                            phi(i,j,k,icomp) = 0.0_rt;
                        }
                    } else {
                        Real tmp = 0.0_rt;
                        for (int m = 1; m < order; ++m) {
                            tmp += phi(i+m*s,j,k,icomp) * coef(m,order-2);
                        }
                        phi(i,j,k,icomp) = tmp;
                        if (inhomog) {
                            phi(i,j,k,icomp) += bcval(i,j,k,icomp)*coef(0,order-2);
                        }
                    }
                }
            }
        }
        break;
    }
    default: {}
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabeclap_apply_bc_y (int side, Box const& box, int blen,
                             Array4<Real> const& phi,
                             Array4<int const> const& mask,
                             Array4<Real const> const& area,
                             BoundCond bct, Real bcl,
                             Array4<Real const> const& bcval,
                             int maxorder, Real dyinv, int inhomog, int icomp) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);
    const int j = lo.y; // boundary cell
    const int s = 1-2*side; // +1 for lo and -1 for hi
    switch (bct) {
    case AMREX_LO_NEUMANN:
    {
        for     (int k = lo.z; k <= hi.z; ++k) {
            for (int i = lo.x; i <= hi.x; ++i) {
                if (mask(i,j,k) == 0 and mask(i,j+s,k) == 1) {
                    phi(i,j,k,icomp) = phi(i,j+s,k,icomp);
                }
            }
        }
        break;
    }
    case AMREX_LO_REFLECT_ODD:
    {
        for     (int k = lo.z; k <= hi.z; ++k) {
            for (int i = lo.x; i <= hi.x; ++i) {
                if (mask(i,j,k) == 0 and mask(i,j+s,k) == 1) {
                    phi(i,j,k,icomp) = -phi(i,j+s,k,icomp);
                }
            }
        }
        break;
    }
    case AMREX_LO_DIRICHLET:
    {
        const int NX = amrex::min(blen+1, maxorder);
        GpuArray<Real,4> x{-bcl * dyinv, 0.5_rt, 1.5_rt, 2.5_rt};
        Array2D<Real, 0, 3, 0, 2> coef{};
        for (int r = 0; r <= maxorder-2; ++r) {
            poly_interp_coeff(-0.5_rt, &x[0], NX, &(coef(0,r)));
        }
        for     (int k = lo.z; k <= hi.z; ++k) {
            for (int i = lo.x; i <= hi.x; ++i) {
                if (mask(i,j,k) == 0 and mask(i,j+s,k) == 1) {
                    int order = 1;
                    for (int r = 0; r <= NX-2; ++r) {
                        if (area(i,j+(1-side)+s*r,k) > 0.0_rt) {
                            ++order;
                        } else {
                            break;
                        }
                    }
                    if (order == 1) {
                        if (inhomog) {
                            phi(i,j,k,icomp) = bcval(i,j,k,icomp);
                        } else {
                            phi(i,j,k,icomp) = 0.0_rt;
                        }
                    } else {
                        Real tmp = 0.0_rt;
                        for (int m = 1; m < order; ++m) {
                            tmp += phi(i,j+m*s,k,icomp) * coef(m,order-2);
                        }
                        phi(i,j,k,icomp) = tmp;
                        if (inhomog) {
                            phi(i,j,k,icomp) += bcval(i,j,k,icomp)*coef(0,order-2);
                        }
                    }
                }
            }
        }
        break;
    }
    default: {}
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebabeclap_apply_bc_z (int side, Box const& box, int blen,
                             Array4<Real> const& phi,
                             Array4<int const> const& mask,
                             Array4<Real const> const& area,
                             BoundCond bct, Real bcl,
                             Array4<Real const> const& bcval,
                             int maxorder, Real dzinv, int inhomog, int icomp) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);
    const int k = lo.z; // boundary cell
    const int s = 1-2*side; // +1 for lo and -1 for hi
    switch (bct) {
    case AMREX_LO_NEUMANN:
    {
        for     (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                if (mask(i,j,k) == 0 and mask(i,j,k+s) == 1) {
                    phi(i,j,k,icomp) = phi(i,j,k+s,icomp);
                }
            }
        }
        break;
    }
    case AMREX_LO_REFLECT_ODD:
    {
        for     (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                if (mask(i,j,k) == 0 and mask(i,j,k+s) == 1) {
                    phi(i,j,k,icomp) = -phi(i,j,k+s,icomp);
                }
            }
        }
        break;
    }
    case AMREX_LO_DIRICHLET:
    {
        const int NX = amrex::min(blen+1, maxorder);
        GpuArray<Real,4> x{-bcl * dzinv, 0.5_rt, 1.5_rt, 2.5_rt};
        Array2D<Real, 0, 3, 0, 2> coef{};
        for (int r = 0; r <= maxorder-2; ++r) {
            poly_interp_coeff(-0.5_rt, &x[0], NX, &(coef(0,r)));
        }
        for     (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                if (mask(i,j,k) == 0 and mask(i,j,k+s) == 1) {
                    int order = 1;
                    for (int r = 0; r <= NX-2; ++r) {
                        if (area(i,j,k+(1-side)+s*r) > 0.0_rt) {
                            ++order;
                        } else {
                            break;
                        }
                    }
                    if (order == 1) {
                        if (inhomog) {
                            phi(i,j,k,icomp) = bcval(i,j,k,icomp);
                        } else {
                            phi(i,j,k,icomp) = 0.0_rt;
                        }
                    } else {
                        Real tmp = 0.0_rt;
                        for (int m = 1; m < order; ++m) {
                            tmp += phi(i,j,k+m*s,icomp) * coef(m,order-2);
                        }
                        phi(i,j,k,icomp) = tmp;
                        if (inhomog) {
                            phi(i,j,k,icomp) += bcval(i,j,k,icomp)*coef(0,order-2);
                        }
                    }
                }
            }
        }
        break;
    }
    default: {}
    }
}

#if (AMREX_SPACEDIM == 2)

// This is the 2D version (i.e. for 6x6 (A^T A) matrix)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void decomp_chol_np6(Array2D<Real,0,5,0,5>& aa)
{
    int neq = 6;

    Real p[neq];
    Real sum1;
    int ising;

    for (int ii = 0; ii < neq; ii++)
    {
        ising = 0;

        for (int jj = ii; jj < neq; jj++)
        {
            sum1 = aa(ii,jj);

            for (int kk = ii-1; kk >= 0; kk--)
            {
                sum1 = sum1 - aa(ii,kk)*aa(jj,kk);
            }
 
            if (ii == jj)
            {
                 if (sum1 <= 0.) 
                 {
                     p[ii] = 0.0;
                     ising = 1;
                 } else {
                     p[ii] = std::sqrt(sum1);
                 }
            } else {
                if (ising == 0) 
                   aa(jj,ii) = sum1 / p[ii];
                else
                   aa(jj,ii) = 0.0;
            }
        }
    }

    for (int ii = 0; ii < neq; ii++)
    {
        for (int jj = ii+1; jj < neq; jj++)
        {
           aa(ii,jj) = 0.0;       // Zero upper triangle
           aa(ii,jj) = aa(jj,ii);  // Zero upper triangle
        }
          
        aa(ii,ii) = p[ii];
    }
}

// This is the 2D version (i.e. for 6x6 (A^T A) matrix)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void  cholsol_np6(Array2D<Real,0,11,0,5>& Amatrix, Array1D<Real,0,5>& b)
{
    int neq = 6;

    Array2D<Real,0,5,0,5> AtA;

    for (int irow = 0; irow < neq; irow++)
        for (int icol = 0; icol < neq; icol++)
            AtA(irow,icol) =  0.0;

    for (int irow = 0; irow < 12; irow++)
    {
         AtA(0,0) += Amatrix(irow,0)*Amatrix(irow,0); // e^T e
         AtA(0,1) += Amatrix(irow,0)*Amatrix(irow,1); // e^T x
         AtA(0,2) += Amatrix(irow,0)*Amatrix(irow,2); // e^T y
         AtA(0,3) += Amatrix(irow,0)*Amatrix(irow,3); // e^T x^2
         AtA(0,4) += Amatrix(irow,0)*Amatrix(irow,4); // e^T x*y
         AtA(0,5) += Amatrix(irow,0)*Amatrix(irow,5); // e^T y^2
         AtA(1,1) += Amatrix(irow,1)*Amatrix(irow,1); // x^T x
         AtA(1,2) += Amatrix(irow,1)*Amatrix(irow,2); // x^T y
         AtA(1,3) += Amatrix(irow,1)*Amatrix(irow,3); // x^T (x^2)
         AtA(1,4) += Amatrix(irow,1)*Amatrix(irow,4); // x^T (xy)
         AtA(1,5) += Amatrix(irow,1)*Amatrix(irow,5); // x^T (y^2)
         AtA(2,2) += Amatrix(irow,2)*Amatrix(irow,2); // y^T y
         AtA(2,3) += Amatrix(irow,2)*Amatrix(irow,3); // y^T (x^2)
         AtA(2,4) += Amatrix(irow,2)*Amatrix(irow,4); // y^T (xy)
         AtA(2,5) += Amatrix(irow,2)*Amatrix(irow,5); // y^T (y^2)
         AtA(3,3) += Amatrix(irow,3)*Amatrix(irow,3); // (x^2)^T (x^2)
         AtA(3,4) += Amatrix(irow,3)*Amatrix(irow,4); // (x^2)^T (x*y)
         AtA(3,5) += Amatrix(irow,3)*Amatrix(irow,5); // (x^2)^T (y^2)
         AtA(4,4) += Amatrix(irow,4)*Amatrix(irow,4); // (x*y)^T (x*y)
         AtA(4,5) += Amatrix(irow,4)*Amatrix(irow,5); // (x*y)^T (y^2)
         AtA(5,5) += Amatrix(irow,5)*Amatrix(irow,5); // (y^2)^T (y^2) 
    }

    for (int irow = 0; irow < neq-1; irow++)
        for (int icol = irow+1; icol < neq; icol++)
           AtA(icol,irow) = AtA(irow,icol);

    decomp_chol_np6(AtA);

    if (AtA(0,0) > 0.)
       b(0) = b(0) / AtA(0,0);
    else
       b(0) = 0.;

    for (int ii = 1; ii < neq; ii++)
    {
       if (AtA(ii,ii) > 0.)
       {
          for (int jj = 0; jj < ii; jj++)
            b(ii) = b(ii) - AtA(ii,jj)*b(jj);

          b(ii) = b(ii) / AtA(ii,ii);
       }
       else
       {
          b(ii) = 0.0;
       }
    }

    if (AtA(neq-1,neq-1) > 0.)
       b(neq-1) = b(neq-1) / AtA(neq-1,neq-1);
    else
       b(neq-1) = 0.0;

    for (int ii = neq-2; ii >= 0; ii--)
    {
       if (AtA(ii,ii) > 0.)
       {
          for (int jj = ii+1; jj < neq; jj++)
            b(ii) = b(ii) - AtA(ii,jj)*b(jj);

          b(ii) = b(ii) / AtA(ii,ii);
       }
       else
       {
          b(ii) = 0.0;
       }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void  cholsol_for_eb(Array2D<Real,0,17,0,5>& Amatrix, Array1D<Real,0,5>& b)
{
    int neq = 6;

    Array2D<Real,0,5,0,5> AtA;

    for (int irow = 0; irow < neq; irow++)
        for (int icol = 0; icol < neq; icol++)
            AtA(irow,icol) =  0.0;

    for (int irow = 0; irow < 18; irow++)
    {
         AtA(0,0) += Amatrix(irow,0)*Amatrix(irow,0); // e^T e
         AtA(0,1) += Amatrix(irow,0)*Amatrix(irow,1); // e^T x
         AtA(0,2) += Amatrix(irow,0)*Amatrix(irow,2); // e^T y
         AtA(0,3) += Amatrix(irow,0)*Amatrix(irow,3); // e^T x^2
         AtA(0,4) += Amatrix(irow,0)*Amatrix(irow,4); // e^T x*y
         AtA(0,5) += Amatrix(irow,0)*Amatrix(irow,5); // e^T y^2
         AtA(1,1) += Amatrix(irow,1)*Amatrix(irow,1); // x^T x
         AtA(1,2) += Amatrix(irow,1)*Amatrix(irow,2); // x^T y
         AtA(1,3) += Amatrix(irow,1)*Amatrix(irow,3); // x^T (x^2)
         AtA(1,4) += Amatrix(irow,1)*Amatrix(irow,4); // x^T (xy)
         AtA(1,5) += Amatrix(irow,1)*Amatrix(irow,5); // x^T (y^2)
         AtA(2,2) += Amatrix(irow,2)*Amatrix(irow,2); // y^T y
         AtA(2,3) += Amatrix(irow,2)*Amatrix(irow,3); // y^T (x^2)
         AtA(2,4) += Amatrix(irow,2)*Amatrix(irow,4); // y^T (xy)
         AtA(2,5) += Amatrix(irow,2)*Amatrix(irow,5); // y^T (y^2)
         AtA(3,3) += Amatrix(irow,3)*Amatrix(irow,3); // (x^2)^T (x^2)
         AtA(3,4) += Amatrix(irow,3)*Amatrix(irow,4); // (x^2)^T (x*y)
         AtA(3,5) += Amatrix(irow,3)*Amatrix(irow,5); // (x^2)^T (y^2)
         AtA(4,4) += Amatrix(irow,4)*Amatrix(irow,4); // (x*y)^T (x*y)
         AtA(4,5) += Amatrix(irow,4)*Amatrix(irow,5); // (x*y)^T (y^2)
         AtA(5,5) += Amatrix(irow,5)*Amatrix(irow,5); // (y^2)^T (y^2) 
    }

    for (int irow = 0; irow < neq-1; irow++)
        for (int icol = irow+1; icol < neq; icol++)
           AtA(icol,irow) = AtA(irow,icol);

    decomp_chol_np6(AtA);

    if (AtA(0,0) > 0.)
       b(0) = b(0) / AtA(0,0);
    else
       b(0) = 0.;

    for (int ii = 1; ii < neq; ii++)
    {
       if (AtA(ii,ii) > 0.)
       {
          for (int jj = 0; jj < ii; jj++)
            b(ii) = b(ii) - AtA(ii,jj)*b(jj);

          b(ii) = b(ii) / AtA(ii,ii);
       }
       else
       {
          b(ii) = 0.0;
       }
    }

    if (AtA(neq-1,neq-1) > 0.)
       b(neq-1) = b(neq-1) / AtA(neq-1,neq-1);
    else
       b(neq-1) = 0.0;

    for (int ii = neq-2; ii >= 0; ii--)
    {
       if (AtA(ii,ii) > 0.)
       {
          for (int jj = ii+1; jj < neq; jj++)
            b(ii) = b(ii) - AtA(ii,jj)*b(jj);

          b(ii) = b(ii) / AtA(ii,ii);
       }
       else
       {
          b(ii) = 0.0;
       }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real grad_x_of_phi_on_centroids(int i,int j,int k,int n,
                                Array4<Real const> const& phi, 
                                Array4<Real const> const& phieb, 
                                Array4<EBCellFlag const> const& flag,
                                Array4<Real const> const& ccent,
                                Array4<Real const> const& bcent, 
                                Array4<Real const> const& apx, 
                                Array4<Real const> const& apy, 
                                Real& yloc_on_xface, 
                                bool is_eb_dirichlet, bool is_eb_inhomog)
{
    Array2D<Real,0,11,0,5> Amatrix;
    Array1D<Real,0, 5    > rhs;

    // Order of column -- first  six are cell centroids: (i,j-1) (i,j) (i,j+1) (i-1,j-1) (i-1,j) (i-1,j+1)
    // Order of column -- second six are   EB centroids: (i,j-1) (i,j) (i,j+1) (i-1,j-1) (i-1,j) (i-1,j+1)

    for (int irow = 0; irow < 12; irow++)
        for (int icol = 0; icol < 6; icol++)
            Amatrix(irow,icol) =  0.0;

    // Columns: [e x y x*x x*y y*y]
    for (int ii = i-1; ii <= i; ii++) // Normal to face
        for (int jj = j-1; jj <= j+1; jj++)  // Tangential to face
        {
            if (!flag(ii,jj,k).isCovered())
            {
                int a_ind  = (jj-(j-1)) + 3*(i-ii);
    
                Real x_off = static_cast<Real>(ii-i) + 0.5;
                Real y_off = static_cast<Real>(jj-j); 
    
                Amatrix(a_ind,0) = 1.0;
                Amatrix(a_ind,1) = x_off + ccent(ii,jj,k,0);
                Amatrix(a_ind,2) = y_off + ccent(ii,jj,k,1) - yloc_on_xface;
                Amatrix(a_ind,3) = Amatrix(a_ind,1) * Amatrix(a_ind,1);
                Amatrix(a_ind,4) = Amatrix(a_ind,1) * Amatrix(a_ind,2);
                Amatrix(a_ind,5) = Amatrix(a_ind,2) * Amatrix(a_ind,2);

                if (!flag(ii,jj,k).isRegular())
                {
                    int a_ind_eb  = a_ind + 6;
                    if (is_eb_dirichlet) {
                        Amatrix(a_ind_eb,0) = 1.0;
                        Amatrix(a_ind_eb,1) = x_off + bcent(ii,jj,k,0);
                        Amatrix(a_ind_eb,2) = y_off + bcent(ii,jj,k,1) - yloc_on_xface; 
                        Amatrix(a_ind_eb,3) = Amatrix(a_ind_eb,1) * Amatrix(a_ind_eb,1);
                        Amatrix(a_ind_eb,4) = Amatrix(a_ind_eb,1) * Amatrix(a_ind_eb,2);
                        Amatrix(a_ind_eb,5) = Amatrix(a_ind_eb,2) * Amatrix(a_ind_eb,2);
                    } else { // Neumann bc's on EB face
                        Real dapx = apx(ii,jj,k)-apx(ii+1,jj,k);
                        Real dapy = apy(ii,jj,k)-apy(ii,jj+1,k);;
                        Real anorm = std::hypot(dapx,dapy);
                        Real norm_x = dapx / anorm;
                        Real norm_y = dapy / anorm;
                        Amatrix(a_ind_eb,0) = 0.0;
                        Amatrix(a_ind_eb,1) = norm_x;
                        Amatrix(a_ind_eb,2) = norm_y;
                        Amatrix(a_ind_eb,3) = 2.0*norm_x*(x_off + bcent(ii,jj,k,1));
                        Amatrix(a_ind_eb,4) = norm_x*(y_off + bcent(ii,jj,k,1) - yloc_on_xface)
                                             +norm_y*(x_off + bcent(ii,jj,k,0));
                        Amatrix(a_ind_eb,5) = 2.0*norm_y*(y_off + bcent(ii,jj,k,1) - yloc_on_xface);
                    }
                }
            }
        }

    // Make the RHS = A^T v
    for (int irow = 0; irow < 6; irow++)
    {
        rhs(irow) = 0.; // Only non-zero when inhomogeneous Dirichlet

        for (int ii = i-1; ii <= i; ii++)  // Normal to face
            for (int jj = j-1; jj <= j+1; jj++)  // Tangential to face
            {
                if (!flag(ii,jj,k).isCovered())
                {
                    int a_ind  = (jj-(j-1)) + 3*(i-ii);
                    rhs(irow) += Amatrix(a_ind  ,irow)*  phi(ii,jj,k,n);

                    if (flag(ii,jj,k).isSingleValued() && 
                        is_eb_dirichlet && is_eb_inhomog)
                            rhs(irow) += Amatrix(a_ind+6,irow)*phieb(ii,jj,k,n);
                }
            }
    }

    cholsol_np6(Amatrix, rhs);

    return rhs(1);     
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real grad_y_of_phi_on_centroids(int i,int j,int k,int n,
                                Array4<Real const> const& phi, 
                                Array4<Real const> const& phieb, 
                                Array4<EBCellFlag const> const& flag,
                                Array4<Real const> const& ccent,
                                Array4<Real const> const& bcent, 
                                Array4<Real const> const& apx, 
                                Array4<Real const> const& apy, 
                                Real& xloc_on_yface,
                                bool is_eb_dirichlet, bool is_eb_inhomog)
{
    Array2D<Real,0,11,0,5> Amatrix;
    Array1D<Real,0, 5    > rhs;

    // Order of column -- first  six are cell centroids: (i-1,j) (i,j) (i+1,j) (i-1,j-1) (i,j-1) (i+1,j-1)
    // Order of column -- second six are   EB centroids: (i-1,j) (i,j) (i+1,j) (i-1,j-1) (i,j-1) (i+1,j-1)

    for (int irow = 0; irow < 12; irow++)
        for (int icol = 0; icol < 6; icol++)
            Amatrix(irow,icol) =  0.0;

    // Columns: [e x y x*x x*y y*y]
    for (int jj = j-1; jj <= j; jj++) // Normal to face
        for (int ii = i-1; ii <= i+1; ii++)  // Tangential to face
        {
            if (!flag(ii,jj,k).isCovered())
            {
                int a_ind  = (ii-(i-1)) + 3*(j-jj);

                Real x_off = static_cast<Real>(ii-i); 
                Real y_off = static_cast<Real>(jj-j) + 0.5;
    
                Amatrix(a_ind,0) = 1.0;
                Amatrix(a_ind,1) = x_off + ccent(ii,jj,k,0) - xloc_on_yface;
                Amatrix(a_ind,2) = y_off + ccent(ii,jj,k,1);
                Amatrix(a_ind,3) = Amatrix(a_ind,1) * Amatrix(a_ind,1);
                Amatrix(a_ind,4) = Amatrix(a_ind,1) * Amatrix(a_ind,2);
                Amatrix(a_ind,5) = Amatrix(a_ind,2) * Amatrix(a_ind,2);

                if (!flag(ii,jj,k).isRegular())
                {
                    int a_ind_eb  = a_ind + 6;
                    if (is_eb_dirichlet) {
                        Amatrix(a_ind_eb,0) = 1.0;
                        Amatrix(a_ind_eb,1) = x_off + bcent(ii,jj,k,0) - xloc_on_yface;
                        Amatrix(a_ind_eb,2) = y_off + bcent(ii,jj,k,1);
                        Amatrix(a_ind_eb,3) = Amatrix(a_ind_eb,1) * Amatrix(a_ind_eb,1);
                        Amatrix(a_ind_eb,4) = Amatrix(a_ind_eb,1) * Amatrix(a_ind_eb,2);
                        Amatrix(a_ind_eb,5) = Amatrix(a_ind_eb,2) * Amatrix(a_ind_eb,2);
                    } else { // Neumann bc's on EB face
                        Real dapx = apx(ii,jj,k)-apx(ii+1,jj,k);
                        Real dapy = apy(ii,jj,k)-apy(ii,jj+1,k);;
                        Real anorm = std::hypot(dapx,dapy);
                        Real norm_x = dapx / anorm;
                        Real norm_y = dapy / anorm;
                        Amatrix(a_ind_eb,0) = 0.0;
                        Amatrix(a_ind_eb,1) = norm_x;
                        Amatrix(a_ind_eb,2) = norm_y;
                        Amatrix(a_ind_eb,3) = 2.0*norm_x*(x_off + bcent(ii,jj,k,0) - xloc_on_yface);
                        Amatrix(a_ind_eb,4) = norm_x*(y_off + bcent(ii,jj,k,1))
                                             +norm_y*(x_off + bcent(ii,jj,k,0) - xloc_on_yface);
                        Amatrix(a_ind_eb,5) = 2.0*norm_y*(y_off + bcent(ii,jj,k,1));
                    }
                }
            }
        }

    // Make the RHS = A^T v
    for (int irow = 0; irow < 6; irow++)
    {
        rhs(irow) = 0.; // Only non-zero when inhomogeneous Dirichlet

        for (int jj = j-1; jj <= j; jj++) // Normal to face
            for (int ii = i-1; ii <= i+1; ii++) // Tangential to face
                if (!flag(ii,jj,k).isCovered())
                {
                    int a_ind  = (ii-(i-1)) + 3*(j-jj);
                    rhs(irow) += Amatrix(a_ind  ,irow)*  phi(ii,jj,k,n);

                    if (flag(ii,jj,k).isSingleValued() && 
                        is_eb_dirichlet && is_eb_inhomog)
                            rhs(irow) += Amatrix(a_ind+6,irow)*phieb(ii,jj,k,n);
                }
    }

    cholsol_np6(Amatrix, rhs);

    return rhs(2);     
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real grad_eb_of_phi_on_centroids(int i,int j,int k,int n,
                                Array4<Real const> const& phi, 
                                Array4<Real const> const& phieb, 
                                Array4<EBCellFlag const> const& flag,
                                Array4<Real const> const& ccent,
                                Array4<Real const> const& bcent,
                                Real& nrmx, Real& nrmy,
                                bool is_eb_inhomog)
{
    Array2D<Real,0,17,0,5> Amatrix;
    Array1D<Real,0, 5    > rhs;

    // Order of column -- first 9 are cell centroids, next 9 are EB centroids

    for (int irow = 0; irow < 18; irow++)
        for (int icol = 0; icol < 6; icol++)
            Amatrix(irow,icol) =  0.0;

    //  Column 0-2: [e x y]
    for (int ii = i-1; ii <= i+1; ii++)
        for (int jj = j-1; jj <= j+1; jj++)
        {

            if (!flag(ii,jj,k).isCovered())
            {
                int a_ind  = (jj-(j-1)) + 3*(ii-(i-1));
    
                Real x_off = static_cast<Real>(ii-i);
                Real y_off = static_cast<Real>(jj-j);
    
                if (flag(i,j,k).isConnected((jj-j),(ii-i),0))
                {
                   Amatrix(a_ind,0) = 1.0;
                   Amatrix(a_ind,1) = x_off + ccent(ii,jj,k,0) - bcent(i,j,k,0);
                   Amatrix(a_ind,2) = y_off + ccent(ii,jj,k,1) - bcent(i,j,k,1);
                }
#if 1
                if (i == ii and j == jj)
                {
                   Amatrix(a_ind,0) *= 2.0;
                   Amatrix(a_ind,1) *= 2.0;
                   Amatrix(a_ind,2) *= 2.0;
                }
#endif

                if (flag(i,j,k).isConnected((jj-j),(ii-i),0) and !flag(ii,jj,k).isRegular())
                {
                    Amatrix(a_ind+9,0) = 1.0;
                    Amatrix(a_ind+9,1) = x_off + bcent(ii,jj,k,0) - bcent(i,j,k,0);
                    Amatrix(a_ind+9,2) = y_off + bcent(ii,jj,k,1) - bcent(i,j,k,1);

#if 1
                    if (i == ii and j == jj)
                        Amatrix(a_ind+9,0) *= 2.0;
#endif
                }
            }
        }

    // Columns 3 : [x*x  x*y  y*y]

    for (int irow = 0; irow < 18; irow++)
    {
       Amatrix(irow,3) =  Amatrix(irow,1) * Amatrix(irow,1);
       Amatrix(irow,4) =  Amatrix(irow,1) * Amatrix(irow,2);
       Amatrix(irow,5) =  Amatrix(irow,2) * Amatrix(irow,2);
    }

    // Make the RHS = A^T v
    for (int irow = 0; irow < 6; irow++)
    {
        rhs(irow) = 0.;

        for (int ii = i-1; ii <= i+1; ii++)
            for (int jj = j-1; jj <= j+1; jj++)
                if (!flag(ii,jj,k).isCovered())
                {
                    int a_ind  = (jj-(j-1)) + 3*(ii-(i-1));
                    rhs(irow) += Amatrix(a_ind,irow) * phi(ii,jj,k,n);
                    if (flag(ii,jj,k).isSingleValued() and is_eb_inhomog)
                        rhs(irow) += Amatrix(a_ind+9,irow)*phieb(ii,jj,k,n);
                }
    }

    cholsol_for_eb(Amatrix, rhs);

    Real dphidn = rhs(1)*nrmx + rhs(2)*nrmy;
    return dphidn;
}

#endif

#if (AMREX_SPACEDIM == 3)

// This is the 3D version (i.e. for 10x10 (A^T A) matrix)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void decomp_chol_np10(Array2D<Real,0,9,0,9>& aa)
{
    int neq = 10;

    Real p[neq];
    Real sum1;
    int ising;

    for (int ii = 0; ii < neq; ii++)
    {
        ising = 0;

        for (int jj = ii; jj < neq; jj++)
        {
            sum1 = aa(ii,jj);

            for (int kk = ii-1; kk >= 0; kk--)
            {
                sum1 = sum1 - aa(ii,kk)*aa(jj,kk);
            }
 
            if (ii == jj)
            {
                 if (sum1 <= 0.) 
                 {
                     p[ii] = 0.0;
                     ising = 1;
                 } else {
                     p[ii] = std::sqrt(sum1);
                 }
            } else {
                if (ising == 0) 
                   aa(jj,ii) = sum1 / p[ii];
                else
                   aa(jj,ii) = 0.0;
            }
        }
    }

    for (int ii = 0; ii < neq; ii++)
    {
        for (int jj = ii+1; jj < neq; jj++)
        {
           aa(ii,jj) = 0.0;       // Zero upper triangle
           aa(ii,jj) = aa(jj,ii);  // Zero upper triangle
        }
          
        aa(ii,ii) = p[ii];
    }
}

// This is the 3D version (i.e. for 10x10 (A^T A) matrix)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void  cholsol_np10(Array2D<Real,0,35,0,9>& Amatrix, Array1D<Real,0,9>& b)
{
    int neq = 10;

    Array2D<Real,0,9,0,9> AtA;

    for (int irow = 0; irow < neq; irow++)
        for (int icol = 0; icol < neq; icol++)
            AtA(irow,icol) =  0.0;

    for (int irow = 0; irow < 36; irow++)
    {
         AtA(0,0) += Amatrix(irow,0)*Amatrix(irow,0); // e^T e
         AtA(0,1) += Amatrix(irow,0)*Amatrix(irow,1); // e^T x
         AtA(0,2) += Amatrix(irow,0)*Amatrix(irow,2); // e^T y
         AtA(0,3) += Amatrix(irow,0)*Amatrix(irow,3); // e^T z
         AtA(0,4) += Amatrix(irow,0)*Amatrix(irow,4); // e^T x^2
         AtA(0,5) += Amatrix(irow,0)*Amatrix(irow,5); // e^T x*y
         AtA(0,6) += Amatrix(irow,0)*Amatrix(irow,6); // e^T y^2
         AtA(0,7) += Amatrix(irow,0)*Amatrix(irow,7); // e^T x*z
         AtA(0,8) += Amatrix(irow,0)*Amatrix(irow,8); // e^T y*z
         AtA(0,9) += Amatrix(irow,0)*Amatrix(irow,9); // e^T z^2

         AtA(1,1) += Amatrix(irow,1)*Amatrix(irow,1); // x^T x
         AtA(1,2) += Amatrix(irow,1)*Amatrix(irow,2); // x^T y
         AtA(1,3) += Amatrix(irow,1)*Amatrix(irow,3); // x^T y
         AtA(1,4) += Amatrix(irow,1)*Amatrix(irow,4); // x^T (x^2)
         AtA(1,5) += Amatrix(irow,1)*Amatrix(irow,5); // x^T (xy)
         AtA(1,6) += Amatrix(irow,1)*Amatrix(irow,6); // x^T (y^2)
         AtA(1,7) += Amatrix(irow,1)*Amatrix(irow,7); // x^T x*z
         AtA(1,8) += Amatrix(irow,1)*Amatrix(irow,8); // x^T y*z
         AtA(1,9) += Amatrix(irow,1)*Amatrix(irow,9); // x^T z^2

         AtA(2,2) += Amatrix(irow,2)*Amatrix(irow,2); // y^T y
         AtA(2,3) += Amatrix(irow,2)*Amatrix(irow,3); // y^T z
         AtA(2,4) += Amatrix(irow,2)*Amatrix(irow,4); // y^T (x^2)
         AtA(2,5) += Amatrix(irow,2)*Amatrix(irow,5); // y^T (xy)
         AtA(2,6) += Amatrix(irow,2)*Amatrix(irow,6); // y^T (y^2)
         AtA(2,7) += Amatrix(irow,2)*Amatrix(irow,7); // y^T x*z
         AtA(2,8) += Amatrix(irow,2)*Amatrix(irow,8); // y^T y*z
         AtA(2,9) += Amatrix(irow,2)*Amatrix(irow,9); // y^T z^2

         AtA(3,3) += Amatrix(irow,3)*Amatrix(irow,3); // z^T z
         AtA(3,4) += Amatrix(irow,3)*Amatrix(irow,4); // z^T (x^2)
         AtA(3,5) += Amatrix(irow,3)*Amatrix(irow,5); // z^T (xy)
         AtA(3,6) += Amatrix(irow,3)*Amatrix(irow,6); // z^T (y^2)
         AtA(3,7) += Amatrix(irow,3)*Amatrix(irow,7); // z^T x*z
         AtA(3,8) += Amatrix(irow,3)*Amatrix(irow,8); // z^T y*z
         AtA(3,9) += Amatrix(irow,3)*Amatrix(irow,9); // z^T z^2

         AtA(4,4) += Amatrix(irow,4)*Amatrix(irow,4); // (x^2)^T (x^2)
         AtA(4,5) += Amatrix(irow,4)*Amatrix(irow,5); // (x^2)^T (xy)
         AtA(4,6) += Amatrix(irow,4)*Amatrix(irow,6); // (x^2)^T (y^2)
         AtA(4,7) += Amatrix(irow,4)*Amatrix(irow,7); // (x^2)^T x*z
         AtA(4,8) += Amatrix(irow,4)*Amatrix(irow,8); // (x^2)^T y*z
         AtA(4,9) += Amatrix(irow,4)*Amatrix(irow,9); // (x^2)^T z^2

         AtA(5,5) += Amatrix(irow,5)*Amatrix(irow,5); // (xy)^T (xy)
         AtA(5,6) += Amatrix(irow,5)*Amatrix(irow,6); // (xy)^T (y^2)
         AtA(5,7) += Amatrix(irow,5)*Amatrix(irow,7); // (xy)^T x*z
         AtA(5,8) += Amatrix(irow,5)*Amatrix(irow,8); // (xy)^T y*z
         AtA(5,9) += Amatrix(irow,5)*Amatrix(irow,9); // (xy)^T z^2

         AtA(6,6) += Amatrix(irow,6)*Amatrix(irow,6); // (y^2)^T (y^2)
         AtA(6,7) += Amatrix(irow,6)*Amatrix(irow,7); // (y^2)^T x*z
         AtA(6,8) += Amatrix(irow,6)*Amatrix(irow,8); // (y^2)^T y*z
         AtA(6,9) += Amatrix(irow,6)*Amatrix(irow,9); // (y^2)^T z^2

         AtA(7,7) += Amatrix(irow,7)*Amatrix(irow,7); // (xz)^T x*z
         AtA(7,8) += Amatrix(irow,7)*Amatrix(irow,8); // (xz)^T y*z
         AtA(7,9) += Amatrix(irow,7)*Amatrix(irow,9); // (xz)^T z^2

         AtA(8,8) += Amatrix(irow,8)*Amatrix(irow,8); // (yz)^T y*z
         AtA(8,9) += Amatrix(irow,8)*Amatrix(irow,9); // (yz)^T z^2

         AtA(9,9) += Amatrix(irow,9)*Amatrix(irow,9); // (z^2)^T z^2
    }

    for (int irow = 0; irow < neq-1; irow++)
        for (int icol = irow+1; icol < neq; icol++)
           AtA(icol,irow) = AtA(irow,icol);

    decomp_chol_np10(AtA);

    if (AtA(0,0) > 0.)
       b(0) = b(0) / AtA(0,0);
    else
       b(0) = 0.;

    for (int ii = 1; ii < neq; ii++)
    {
       if (AtA(ii,ii) > 0.)
       {
          for (int jj = 0; jj < ii; jj++)
            b(ii) = b(ii) - AtA(ii,jj)*b(jj);

          b(ii) = b(ii) / AtA(ii,ii);
       }
       else
       {
          b(ii) = 0.0;
       }
    }

    if (AtA(neq-1,neq-1) > 0.)
       b(neq-1) = b(neq-1) / AtA(neq-1,neq-1);
    else
       b(neq-1) = 0.0;

    for (int ii = neq-2; ii >= 0; ii--)
    {
       if (AtA(ii,ii) > 0.)
       {
          for (int jj = ii+1; jj < neq; jj++)
            b(ii) = b(ii) - AtA(ii,jj)*b(jj);

          b(ii) = b(ii) / AtA(ii,ii);
       }
       else
       {
          b(ii) = 0.0;
       }
    }
}

// This is the 3D version (i.e. for 10x10 (A^T A) matrix) but for A = 54 x 10
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void  cholsol_for_eb(Array2D<Real,0,53,0,9>& Amatrix, Array1D<Real,0,9>& b)
{
    int neq = 10;

    Array2D<Real,0,9,0,9> AtA;

    for (int irow = 0; irow < neq; irow++)
        for (int icol = 0; icol < neq; icol++)
            AtA(irow,icol) =  0.0;

    for (int irow = 0; irow < 54; irow++)
    {
         AtA(0,0) += Amatrix(irow,0)*Amatrix(irow,0); // e^T e
         AtA(0,1) += Amatrix(irow,0)*Amatrix(irow,1); // e^T x
         AtA(0,2) += Amatrix(irow,0)*Amatrix(irow,2); // e^T y
         AtA(0,3) += Amatrix(irow,0)*Amatrix(irow,3); // e^T z
         AtA(0,4) += Amatrix(irow,0)*Amatrix(irow,4); // e^T x^2
         AtA(0,5) += Amatrix(irow,0)*Amatrix(irow,5); // e^T x*y
         AtA(0,6) += Amatrix(irow,0)*Amatrix(irow,6); // e^T y^2
         AtA(0,7) += Amatrix(irow,0)*Amatrix(irow,7); // e^T x*z
         AtA(0,8) += Amatrix(irow,0)*Amatrix(irow,8); // e^T y*z
         AtA(0,9) += Amatrix(irow,0)*Amatrix(irow,9); // e^T z^2

         AtA(1,1) += Amatrix(irow,1)*Amatrix(irow,1); // x^T x
         AtA(1,2) += Amatrix(irow,1)*Amatrix(irow,2); // x^T y
         AtA(1,3) += Amatrix(irow,1)*Amatrix(irow,3); // x^T y
         AtA(1,4) += Amatrix(irow,1)*Amatrix(irow,4); // x^T (x^2)
         AtA(1,5) += Amatrix(irow,1)*Amatrix(irow,5); // x^T (xy)
         AtA(1,6) += Amatrix(irow,1)*Amatrix(irow,6); // x^T (y^2)
         AtA(1,7) += Amatrix(irow,1)*Amatrix(irow,7); // x^T x*z
         AtA(1,8) += Amatrix(irow,1)*Amatrix(irow,8); // x^T y*z
         AtA(1,9) += Amatrix(irow,1)*Amatrix(irow,9); // x^T z^2

         AtA(2,2) += Amatrix(irow,2)*Amatrix(irow,2); // y^T y
         AtA(2,3) += Amatrix(irow,2)*Amatrix(irow,3); // y^T z
         AtA(2,4) += Amatrix(irow,2)*Amatrix(irow,4); // y^T (x^2)
         AtA(2,5) += Amatrix(irow,2)*Amatrix(irow,5); // y^T (xy)
         AtA(2,6) += Amatrix(irow,2)*Amatrix(irow,6); // y^T (y^2)
         AtA(2,7) += Amatrix(irow,2)*Amatrix(irow,7); // y^T x*z
         AtA(2,8) += Amatrix(irow,2)*Amatrix(irow,8); // y^T y*z
         AtA(2,9) += Amatrix(irow,2)*Amatrix(irow,9); // y^T z^2

         AtA(3,3) += Amatrix(irow,3)*Amatrix(irow,3); // z^T z
         AtA(3,4) += Amatrix(irow,3)*Amatrix(irow,4); // z^T (x^2)
         AtA(3,5) += Amatrix(irow,3)*Amatrix(irow,5); // z^T (xy)
         AtA(3,6) += Amatrix(irow,3)*Amatrix(irow,6); // z^T (y^2)
         AtA(3,7) += Amatrix(irow,3)*Amatrix(irow,7); // z^T x*z
         AtA(3,8) += Amatrix(irow,3)*Amatrix(irow,8); // z^T y*z
         AtA(3,9) += Amatrix(irow,3)*Amatrix(irow,9); // z^T z^2

         AtA(4,4) += Amatrix(irow,4)*Amatrix(irow,4); // (x^2)^T (x^2)
         AtA(4,5) += Amatrix(irow,4)*Amatrix(irow,5); // (x^2)^T (xy)
         AtA(4,6) += Amatrix(irow,4)*Amatrix(irow,6); // (x^2)^T (y^2)
         AtA(4,7) += Amatrix(irow,4)*Amatrix(irow,7); // (x^2)^T x*z
         AtA(4,8) += Amatrix(irow,4)*Amatrix(irow,8); // (x^2)^T y*z
         AtA(4,9) += Amatrix(irow,4)*Amatrix(irow,9); // (x^2)^T z^2

         AtA(5,5) += Amatrix(irow,5)*Amatrix(irow,5); // (xy)^T (xy)
         AtA(5,6) += Amatrix(irow,5)*Amatrix(irow,6); // (xy)^T (y^2)
         AtA(5,7) += Amatrix(irow,5)*Amatrix(irow,7); // (xy)^T x*z
         AtA(5,8) += Amatrix(irow,5)*Amatrix(irow,8); // (xy)^T y*z
         AtA(5,9) += Amatrix(irow,5)*Amatrix(irow,9); // (xy)^T z^2

         AtA(6,6) += Amatrix(irow,6)*Amatrix(irow,6); // (y^2)^T (y^2)
         AtA(6,7) += Amatrix(irow,6)*Amatrix(irow,7); // (y^2)^T x*z
         AtA(6,8) += Amatrix(irow,6)*Amatrix(irow,8); // (y^2)^T y*z
         AtA(6,9) += Amatrix(irow,6)*Amatrix(irow,9); // (y^2)^T z^2

         AtA(7,7) += Amatrix(irow,7)*Amatrix(irow,7); // (xz)^T x*z
         AtA(7,8) += Amatrix(irow,7)*Amatrix(irow,8); // (xz)^T y*z
         AtA(7,9) += Amatrix(irow,7)*Amatrix(irow,9); // (xz)^T z^2

         AtA(8,8) += Amatrix(irow,8)*Amatrix(irow,8); // (yz)^T y*z
         AtA(8,9) += Amatrix(irow,8)*Amatrix(irow,9); // (yz)^T z^2

         AtA(9,9) += Amatrix(irow,9)*Amatrix(irow,9); // (z^2)^T z^2
    }

    for (int irow = 0; irow < neq-1; irow++)
        for (int icol = irow+1; icol < neq; icol++)
           AtA(icol,irow) = AtA(irow,icol);

    decomp_chol_np10(AtA);

    if (AtA(0,0) > 0.)
       b(0) = b(0) / AtA(0,0);
    else
       b(0) = 0.;

    for (int ii = 1; ii < neq; ii++)
    {
       if (AtA(ii,ii) > 0.)
       {
          for (int jj = 0; jj < ii; jj++)
            b(ii) = b(ii) - AtA(ii,jj)*b(jj);

          b(ii) = b(ii) / AtA(ii,ii);
       }
       else
       {
          b(ii) = 0.0;
       }
    }

    if (AtA(neq-1,neq-1) > 0.)
       b(neq-1) = b(neq-1) / AtA(neq-1,neq-1);
    else
       b(neq-1) = 0.0;

    for (int ii = neq-2; ii >= 0; ii--)
    {
       if (AtA(ii,ii) > 0.)
       {
          for (int jj = ii+1; jj < neq; jj++)
            b(ii) = b(ii) - AtA(ii,jj)*b(jj);

          b(ii) = b(ii) / AtA(ii,ii);
       }
       else
       {
          b(ii) = 0.0;
       }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real grad_x_of_phi_on_centroids(int i,int j,int k,int n,
                                Array4<Real const> const& phi, 
                                Array4<Real const> const& phieb, 
                                Array4<EBCellFlag const> const& flag,
                                Array4<Real const> const& ccent,
                                Array4<Real const> const& bcent, 
                                Array4<Real const> const& apx, 
                                Array4<Real const> const& apy, 
                                Array4<Real const> const& apz, 
                                Real& yloc_on_xface, Real& zloc_on_xface,
                                bool is_eb_dirichlet, bool is_eb_inhomog)
{
    Array2D<Real,0,35,0,9> Amatrix;
    Array1D<Real,0, 9    > rhs;

    // Order of column -- first 9 are cell centroids, second 9 are EB centroids

    for (int irow = 0; irow < 36; irow++)
        for (int icol = 0; icol < 6; icol++)
            Amatrix(irow,icol) =  0.0;

    //  Columns 0-3: [e x y z]
    for (int ii = i-1; ii <= i; ii++) // Normal to face
      for (int kk = k-1; kk <= k+1; kk++)  // Tangential to face
         for (int jj = j-1; jj <= j+1; jj++)  // Tangential to face
            if (!flag(ii,jj,kk).isCovered())
            {
                int a_ind  = (jj-(j-1)) + 3*(kk-(k-1)) + 9*(i-ii);
    
                Real x_off = static_cast<Real>(ii-i) + 0.5;
                Real y_off = static_cast<Real>(jj-j); 
                Real z_off = static_cast<Real>(jj-j); 
    
                Amatrix(a_ind,0) = 1.0;
                Amatrix(a_ind,1) = x_off + ccent(ii,jj,k,0);
                Amatrix(a_ind,2) = y_off + ccent(ii,jj,k,1) - yloc_on_xface;
                Amatrix(a_ind,3) = z_off + ccent(ii,jj,k,2) - zloc_on_xface;

                if (!flag(ii,jj,k).isRegular())
                {
                    Amatrix(a_ind+6,0) = 1.0;
                    Amatrix(a_ind+6,1) = x_off + bcent(ii,jj,k,0);
                    Amatrix(a_ind+6,2) = y_off + bcent(ii,jj,k,1) - yloc_on_xface;
                    Amatrix(a_ind+6,3) = z_off + bcent(ii,jj,k,2) - zloc_on_xface;
                }
            }

    // Columns 4-9 : [x*x  x*y  y*y  x*z  y*z  z*z]
    for (int irow = 0; irow < 12; irow++)
    {
       Amatrix(irow,4) =  Amatrix(irow,1) * Amatrix(irow,1);
       Amatrix(irow,5) =  Amatrix(irow,1) * Amatrix(irow,2);
       Amatrix(irow,6) =  Amatrix(irow,2) * Amatrix(irow,2);
       Amatrix(irow,7) =  Amatrix(irow,1) * Amatrix(irow,3);
       Amatrix(irow,8) =  Amatrix(irow,2) * Amatrix(irow,3);
       Amatrix(irow,9) =  Amatrix(irow,3) * Amatrix(irow,3);
    }

    // Make the RHS = A^T v
    for (int irow = 0; irow < 10; irow++)
    {
        rhs(irow) = 0.;

        for (int ii = i-1; ii <= i; ii++)  // Normal to face
          for (int kk = k-1; kk <= k+1; kk++)  // Tangential to face
           for (int jj = j-1; jj <= j+1; jj++) // Tangential to face
               if (!flag(ii,jj,kk).isCovered())
               {
                    int a_ind  = (jj-(j-1)) + 3*(kk-(k-1)) + 9*(i-ii);

                    rhs(irow) += Amatrix(a_ind,irow)*  phi(ii,jj,kk,n);

                    if (flag(ii,jj,kk).isSingleValued() and is_eb_inhomog)  
                        rhs(irow) += Amatrix(a_ind+18,irow)*phieb(ii,jj,kk,n);
               }
    }

    cholsol_np10(Amatrix, rhs);

    return rhs(1);     
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real grad_y_of_phi_on_centroids(int i,int j,int k,int n,
                                Array4<Real const> const& phi, 
                                Array4<Real const> const& phieb, 
                                Array4<EBCellFlag const> const& flag,
                                Array4<Real const> const& ccent,
                                Array4<Real const> const& bcent, 
                                Array4<Real const> const& apx, 
                                Array4<Real const> const& apy, 
                                Array4<Real const> const& apz, 
                                Real& xloc_on_yface, Real& zloc_on_yface,
                                bool is_eb_dirichlet, bool is_eb_inhomog)
{
    Array2D<Real,0,35,0,9> Amatrix;
    Array1D<Real,0, 9    > rhs;

    // Order of column -- first 9 are cell centroids, second 9 are EB centroids

    for (int irow = 0; irow < 36; irow++)
        for (int icol = 0; icol < 10; icol++)
            Amatrix(irow,icol) =  0.0;

    //  Columns 0-2: [e x y]
    for (int jj = j-1; jj <= j; jj++) // Normal to face
      for (int kk = k-1; kk <= k+1; kk++)  // Tangential to face
        for (int ii = i-1; ii <= i+1; ii++)  // Tangential to face
            if (!flag(ii,jj,k).isCovered())
            {
                int a_ind  = (ii-(i-1)) + 3*(kk-(k-1)) + 9*(j-jj);

                Real x_off = static_cast<Real>(ii-i); 
                Real y_off = static_cast<Real>(jj-j) + 0.5;
                Real z_off = static_cast<Real>(kk-k);
    
                Amatrix(a_ind,0) = 1.0;
                Amatrix(a_ind,1) = x_off + ccent(ii,jj,k,0) - xloc_on_yface;
                Amatrix(a_ind,2) = y_off + ccent(ii,jj,k,1);
                Amatrix(a_ind,3) = z_off + ccent(ii,jj,k,2) - zloc_on_yface;

                if (flag(ii,jj,k).isSingleValued())
                {
                    Amatrix(a_ind+9,0) = 1.0;
                    Amatrix(a_ind+9,1) = x_off + bcent(ii,jj,k,0) - xloc_on_yface;
                    Amatrix(a_ind+9,2) = y_off + bcent(ii,jj,k,1);
                    Amatrix(a_ind+9,3) = z_off + bcent(ii,jj,k,2) - zloc_on_yface;
                }
            }

    // Columns 4-9 : [x*x  x*y  y*y  x*z  y*z  z*z]
    for (int irow = 0; irow < 12; irow++)
    {
       Amatrix(irow,4) =  Amatrix(irow,1) * Amatrix(irow,1);
       Amatrix(irow,5) =  Amatrix(irow,1) * Amatrix(irow,2);
       Amatrix(irow,6) =  Amatrix(irow,2) * Amatrix(irow,2);
       Amatrix(irow,7) =  Amatrix(irow,1) * Amatrix(irow,3);
       Amatrix(irow,8) =  Amatrix(irow,2) * Amatrix(irow,3);
       Amatrix(irow,9) =  Amatrix(irow,3) * Amatrix(irow,3);
    }

    // Make the RHS = A^T v
    for (int irow = 0; irow < 10; irow++)
    {
        rhs(irow) = 0.;

        for (int jj = j-1; jj <= j; jj++) // Normal to face
           for (int kk = k-1; kk <= k+1; kk++)  // Tangential to face
             for (int ii = i-1; ii <= i+1; ii++) // Tangential to face
                if (!flag(ii,jj,kk).isCovered())
                {
                    int a_ind  = (ii-(i-1)) + 3*(kk-(k-1)) + 9*(j-jj);

                    rhs(irow) += Amatrix(a_ind,irow)*  phi(ii,jj,kk,n);

                    if (flag(ii,jj,kk).isSingleValued() and is_eb_inhomog)  
                        rhs(irow) += Amatrix(a_ind+18,irow)*phieb(ii,jj,kk,n);
                }
    }

    cholsol_np10(Amatrix, rhs);

    return rhs(2);     
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real grad_z_of_phi_on_centroids(int i,int j,int k,int n,
                                Array4<Real const> const& phi, 
                                Array4<Real const> const& phieb, 
                                Array4<EBCellFlag const> const& flag,
                                Array4<Real const> const& ccent,
                                Array4<Real const> const& bcent, 
                                Array4<Real const> const& apx, 
                                Array4<Real const> const& apy, 
                                Array4<Real const> const& apz, 
                                Real& xloc_on_zface, Real& yloc_on_zface,
                                bool is_eb_dirichlet, bool is_eb_inhomog)
{
    Array2D<Real,0,35,0,9> Amatrix;
    Array1D<Real,0, 9    > rhs;

    // Order of column -- first 9 are cell centroids, second 9 are EB centroids

    for (int irow = 0; irow < 36; irow++)
        for (int icol = 0; icol < 6; icol++)
            Amatrix(irow,icol) =  0.0;

    //  Columns 0-3: [e x y z]
    for (int kk = k-1; kk <= k; kk++) // Normal to face
    {
        for (int jj = j-1; jj <= j+1; jj++)  // Tangential to face
         for (int ii = i-1; ii <= i+1; ii++)  // Tangential to face
         {
            if (!flag(ii,jj,kk).isCovered())
            {
                int a_ind  = (ii-(i-1)) + 3*(jj-(j-1)) + 9*(k-kk);

                Real x_off = static_cast<Real>(ii-i); 
                Real y_off = static_cast<Real>(jj-j); 
                Real z_off = static_cast<Real>(kk-k) + 0.5;
    
                Amatrix(a_ind,0) = 1.0;
                Amatrix(a_ind,1) = x_off + ccent(ii,jj,kk,0) - xloc_on_zface;
                Amatrix(a_ind,2) = y_off + ccent(ii,jj,kk,1) - yloc_on_zface;
                Amatrix(a_ind,3) = z_off + ccent(ii,jj,kk,2);

                if (!flag(ii,jj,kk).isRegular())
                {
                    Amatrix(a_ind+10,0) = 1.0;
                    Amatrix(a_ind+10,1) = x_off + bcent(ii,jj,kk,0) - xloc_on_zface;
                    Amatrix(a_ind+10,2) = y_off + bcent(ii,jj,kk,1) - yloc_on_zface;
                    Amatrix(a_ind+10,3) = z_off + bcent(ii,jj,kk,2);
                }
            }
         }
    }

    // Columns 4-9 : [x*x  x*y  y*y  x*z  y*z  z*z]
    for (int irow = 0; irow < 12; irow++)
    {
       Amatrix(irow,4) =  Amatrix(irow,1) * Amatrix(irow,1);
       Amatrix(irow,5) =  Amatrix(irow,1) * Amatrix(irow,2);
       Amatrix(irow,6) =  Amatrix(irow,2) * Amatrix(irow,2);
       Amatrix(irow,7) =  Amatrix(irow,1) * Amatrix(irow,3);
       Amatrix(irow,8) =  Amatrix(irow,2) * Amatrix(irow,3);
       Amatrix(irow,9) =  Amatrix(irow,3) * Amatrix(irow,3);
    }

    // Make the RHS = A^T v
    for (int irow = 0; irow < 10; irow++)
    {
        rhs(irow) = 0.;

        for (int kk = k-1; kk <= k; kk++) // Normal to face
          for (int jj = j-1; jj <= j+1; jj++)  // Tangential to face
             for (int ii = i-1; ii <= i+1; ii++) // Tangential to face
                if (!flag(ii,jj,kk).isCovered())
                {
                    int a_ind  = (ii-(i-1)) + 3*(jj-(j-1)) + 9*(k-kk);

                    rhs(irow) += Amatrix(a_ind,irow)*  phi(ii,jj,kk,n);

                    if (flag(ii,jj,kk).isSingleValued() and is_eb_inhomog)
                        rhs(irow) += Amatrix(a_ind+18,irow)*phieb(ii,jj,kk,n);
                }
    }

    cholsol_np10(Amatrix, rhs);

    return rhs(3);     
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real grad_eb_of_phi_on_centroids(int i,int j,int k,int n,
                                 Array4<Real const> const& phi, 
                                 Array4<Real const> const& phieb, 
                                 Array4<EBCellFlag const> const& flag,
                                 Array4<Real const> const& ccent,
                                 Array4<Real const> const& bcent,
                                 Real& nrmx, Real& nrmy, Real& nrmz,
                                 bool is_eb_inhomog)
{
    Array2D<Real,0,53,0,9> Amatrix;
    Array1D<Real,0, 9    > rhs;

    // Order of column -- first 27 are cell centroids, second 27 are EB centroids

    for (int irow = 0; irow < 54; irow++)
        for (int icol = 0; icol < 6; icol++)
            Amatrix(irow,icol) =  0.0;

    //  Columns 0-3: [e x y z]
    for (int kk = k-1; kk <= k+1; kk++)
      for (int jj = j-1; jj <= j+1; jj++)
         for (int ii = i-1; ii <= i+1; ii++)
         {
            if (!flag(ii,jj,kk).isCovered())
            {
                int a_ind  = (ii-(i-1)) + 3*(jj-(j-1)) + 9*(kk-(k-1));

                Real x_off = static_cast<Real>(ii-i) - bcent(i,j,k,0);
                Real y_off = static_cast<Real>(jj-j) - bcent(i,j,k,1);
                Real z_off = static_cast<Real>(kk-k) - bcent(i,j,k,2);
    
                Amatrix(a_ind,0) = 1.0;
                Amatrix(a_ind,1) = x_off + ccent(ii,jj,kk,0);
                Amatrix(a_ind,2) = y_off + ccent(ii,jj,kk,1);
                Amatrix(a_ind,3) = z_off + ccent(ii,jj,kk,2);

                if (!flag(ii,jj,kk).isRegular())
                {
                    Amatrix(a_ind+27,0) = 1.0;
                    Amatrix(a_ind+27,1) = x_off + bcent(ii,jj,kk,0);
                    Amatrix(a_ind+27,2) = y_off + bcent(ii,jj,kk,1);
                    Amatrix(a_ind+27,3) = z_off + bcent(ii,jj,kk,2);
                }
            }
         }

    // Columns 4-9 : [x*x  x*y  y*y  x*z  y*z  z*z]
    for (int irow = 0; irow < 54; irow++)
    {
       Amatrix(irow,4) =  Amatrix(irow,1) * Amatrix(irow,1);
       Amatrix(irow,5) =  Amatrix(irow,1) * Amatrix(irow,2);
       Amatrix(irow,6) =  Amatrix(irow,2) * Amatrix(irow,2);
       Amatrix(irow,7) =  Amatrix(irow,1) * Amatrix(irow,3);
       Amatrix(irow,8) =  Amatrix(irow,2) * Amatrix(irow,3);
       Amatrix(irow,9) =  Amatrix(irow,3) * Amatrix(irow,3);
    }

    // Make the RHS = A^T v
    for (int irow = 0; irow < 10; irow++)
    {
        rhs(irow) = 0.;

        for (int kk = k-1; kk <= k+1; kk++)
          for (int jj = j-1; jj <= j+1; jj++)
             for (int ii = i-1; ii <= i+1; ii++)
                if (!flag(ii,jj,kk).isCovered())
                {
                    int a_ind  = (ii-(i-1)) + 3*(jj-(j-1)) + 9*(kk-(k-1));

                    rhs(irow) += Amatrix(a_ind,irow)*  phi(ii,jj,kk,n);

                    if (flag(ii,jj,kk).isSingleValued() and is_eb_inhomog)
                        rhs(irow) += Amatrix(a_ind+27,irow)*phieb(ii,jj,kk,n);
                }
    }

    cholsol_for_eb(Amatrix, rhs);

    Real dphidn = rhs(1)*nrmx + rhs(2)*nrmy + rhs(3)*nrmz;
    return dphidn;
}

#endif

}

#endif

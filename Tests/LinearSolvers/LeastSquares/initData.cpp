#include "MyTest.H"
#include <AMReX_EB2.H>
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void
MyTest::initData ()
{
    int nlevels = max_level + 1;
    dmap.resize(nlevels);
    factory.resize(nlevels);
    phi.resize(nlevels);
    phieb.resize(nlevels);
    grad_x.resize(nlevels);
    grad_x_analytic.resize(nlevels);
    grad_y.resize(nlevels);
    grad_y_analytic.resize(nlevels);
    grad_z.resize(nlevels);
    grad_z_analytic.resize(nlevels);
    grad_eb.resize(nlevels);
    grad_eb_analytic.resize(nlevels);
    lap_analytic.resize(nlevels);
    ccentr.resize(nlevels);
    rhs.resize(nlevels);
    acoef.resize(nlevels);
    bcoef.resize(nlevels);
    bcoef_eb.resize(nlevels);

    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        dmap[ilev].define(grids[ilev]);
        const EB2::IndexSpace& eb_is = EB2::IndexSpace::top();
        const EB2::Level& eb_level = eb_is.getLevel(geom[ilev]);
        factory[ilev].reset(new EBFArrayBoxFactory(eb_level, geom[ilev], grids[ilev], dmap[ilev],
                                                   {2,2,2}, EBSupport::full));

        phi[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, 1, MFInfo(), *factory[ilev]);
        phieb[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, 1, MFInfo(), *factory[ilev]);
        grad_x[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, 1, MFInfo(), *factory[ilev]);
        grad_x_analytic[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, 1, MFInfo(), *factory[ilev]);
        grad_y[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, 1, MFInfo(), *factory[ilev]);
        grad_y_analytic[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, 1, MFInfo(), *factory[ilev]);
        grad_z[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, 1, MFInfo(), *factory[ilev]);
        grad_z_analytic[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, 1, MFInfo(), *factory[ilev]);
        grad_eb[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, 1, MFInfo(), *factory[ilev]);
        grad_eb_analytic[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, 1, MFInfo(), *factory[ilev]);
        lap_analytic[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, 1, MFInfo(), *factory[ilev]);
        ccentr[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, 1, MFInfo(), *factory[ilev]);
        rhs[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, 0, MFInfo(), *factory[ilev]);
        acoef[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, 0, MFInfo(), *factory[ilev]);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            bcoef[ilev][idim].define(amrex::convert(grids[ilev],IntVect::TheDimensionVector(idim)),
                                     dmap[ilev], AMREX_SPACEDIM, 0, MFInfo(), *factory[ilev]);
        }
        if (eb_is_dirichlet) {
            bcoef_eb[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, 0, MFInfo(), *factory[ilev]);
            bcoef_eb[ilev].setVal(1.0);
        }

        phi[ilev].setVal(0.0);
        phieb[ilev].setVal(0.0);
        grad_x[ilev].setVal(1e40);
        grad_x_analytic[ilev].setVal(1e40);
        grad_y[ilev].setVal(1e40);
        grad_y_analytic[ilev].setVal(1e40);
        grad_z[ilev].setVal(1e40);
        grad_z_analytic[ilev].setVal(1e40);
        grad_eb[ilev].setVal(1e40);
        grad_eb_analytic[ilev].setVal(1e40);
        lap_analytic[ilev].setVal(1e40);
        ccentr[ilev].setVal(0.0);
        rhs[ilev].setVal(0.0);
        acoef[ilev].setVal(0.0);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            bcoef[ilev][idim].setVal(1.0);
        }

        const auto dx = geom[ilev].CellSizeArray();

        const Real pi = 4.0*std::atan(1.0);

        for (MFIter mfi(phi[ilev]); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.fabbox();
            Array4<Real> const& fab = phi[ilev].array(mfi);
            Array4<Real> const& fab_gx = grad_x_analytic[ilev].array(mfi);
            Array4<Real> const& fab_gy = grad_y_analytic[ilev].array(mfi);
            Array4<Real> const& fab_gz = grad_z_analytic[ilev].array(mfi);
            Array4<Real> const& fab_eb = grad_eb_analytic[ilev].array(mfi);
            Array4<Real> const& fab_lap = lap_analytic[ilev].array(mfi);

            const FabArray<EBCellFlagFab>* flags = &(factory[ilev]->getMultiEBCellFlagFab());
            Array4<EBCellFlag const> const& flag = flags->const_array(mfi);

            if (use_poiseuille_1d) {
               Array4<Real const> const& ccent = (factory[ilev]->getCentroid()).array(mfi);
               Array4<Real const> const& fcx   = (factory[ilev]->getFaceCent())[0]->const_array(mfi);
               Array4<Real const> const& fcy   = (factory[ilev]->getFaceCent())[1]->const_array(mfi);
               Array4<Real const> const& apx   = (factory[ilev]->getAreaFrac())[0]->const_array(mfi);
               Array4<Real const> const& apy   = (factory[ilev]->getAreaFrac())[1]->const_array(mfi);
               Array4<Real const> const& norm  = (factory[ilev]->getBndryNormal()).array(mfi);
               Array4<Real const> const& bcent = (factory[ilev]->getBndryCent()).array(mfi);
#if (AMREX_SPACEDIM > 2)
               Array4<Real const> const& fcz   = (factory[ilev]->getFaceCent())[2]->const_array(mfi);
               Array4<Real const> const& apz   = (factory[ilev]->getAreaFrac())[2]->const_array(mfi);
#endif

               amrex::ParallelFor(bx,
               [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
               {
#if (AMREX_SPACEDIM == 2)
                   Real H = poiseuille_1d_height;
                   Real t = (poiseuille_1d_rotation/180.)*M_PI;
                   
                   Real a = std::tan(t);
                   Real b = -1.0;
                   Real c = poiseuille_1d_pt_on_top_wall[1] - std::tan(t)*poiseuille_1d_pt_on_top_wall[0];
                     

                   Real rx = (i+0.5+ccent(i,j,k,0)) * dx[0];
                   Real ry = (j+0.5+ccent(i,j,k,1)) * dx[1];

                   auto d = std::fabs(a*rx + b*ry + c)/std::sqrt(a*a + b*b);

                   auto phi_mag = (!flag(i,j,k).isCovered()) ? d * (H - d) : 0.0;
                   fab(i,j,k,0) = phi_mag * std::cos(t);
                   fab(i,j,k,1) = phi_mag * std::sin(t);

                   if( flag(i,j,k).isCovered()) {
                     fab_gx(i,j,k,0) = 0.0;
                     fab_gx(i,j,k,1) = 0.0;
                     fab_gy(i,j,k,0) = 0.0;
                     fab_gy(i,j,k,1) = 0.0;
                     fab_lap(i,j,k,0) = 0.0;
                     fab_lap(i,j,k,1) = 0.0;
                   }
                   else {
                     Real rxl = i * dx[0];
                     Real ryl = (j+0.5+fcx(i,j,k,0)) * dx[1];
                     Real fac = (H - 2*(a*rxl+b*ryl+c)/(std::sqrt(a*a + b*b)));
                     fab_gx(i,j,k,0) = (apx(i,j,k) == 0.0) ? 0.0 : (a*std::cos(t)/std::sqrt(a*a + b*b)) * fac * dx[0];
                     fab_gx(i,j,k,1) = (apx(i,j,k) == 0.0) ? 0.0 : (a*std::sin(t)/std::sqrt(a*a + b*b)) * fac * dx[0];

                     rxl = (i+0.5+fcy(i,j,k,0)) * dx[0];
                     ryl = j * dx[1];
                     fac = (H - 2*(a*rxl+b*ryl+c)/(std::sqrt(a*a + b*b)));
                     fab_gy(i,j,k,0) = (apy(i,j,k) == 0.0) ? 0.0 : (b*std::cos(t)/std::sqrt(a*a + b*b)) * fac * dx[1];
                     fab_gy(i,j,k,1) = (apy(i,j,k) == 0.0) ? 0.0 : (b*std::sin(t)/std::sqrt(a*a + b*b)) * fac * dx[1];

                     fab_lap(i,j,k,0) = -2.0*a*std::cos(t)/std::sqrt(a*a + b*b) - 2.0*b*std::cos(t)/std::sqrt(a*a + b*b);

                     fab_lap(i,j,k,1) = -2.0*a*std::sin(t)/std::sqrt(a*a + b*b) - 2.0*b*std::sin(t)/std::sqrt(a*a + b*b);
                   }

                   if(flag(i,j,k).isSingleValued()) {
                     Real rxeb = (i+0.5+bcent(i,j,k,0)) * dx[0];
                     Real ryeb = (j+0.5+bcent(i,j,k,1)) * dx[1];
                     Real fac = (H - 2*(a*rxeb+b*ryeb+c)/(std::sqrt(a*a + b*b)));
                     Real dudx = (a*std::cos(t)/std::sqrt(a*a + b*b)) * fac * dx[0];
                     Real dvdx = (a*std::sin(t)/std::sqrt(a*a + b*b)) * fac * dx[0];
                     Real dudy = (b*std::cos(t)/std::sqrt(a*a + b*b)) * fac * dx[1];
                     Real dvdy = (b*std::sin(t)/std::sqrt(a*a + b*b)) * fac * dx[1];

                     fab_eb(i,j,k,0) = dudx*norm(i,j,k,0) + dudy*norm(i,j,k,1);
                     fab_eb(i,j,k,1) = dvdx*norm(i,j,k,0) + dvdy*norm(i,j,k,1);
                   }
#else
                   if(poiseuille_1d_askew) { //3D askew
                      Real H = poiseuille_1d_height;
                      int nfdir = poiseuille_1d_no_flow_dir;
                      Real alpha = (poiseuille_1d_askew_rotation[0]/180.)*M_PI;
                      Real gamma = (poiseuille_1d_askew_rotation[1]/180.)*M_PI;

                      Real a = std::sin(gamma);
                      Real b = -std::cos(alpha)*std::cos(gamma);
                      Real c = std::sin(alpha);
                      Real d = -a*poiseuille_1d_pt_on_top_wall[0] 
                               -b*poiseuille_1d_pt_on_top_wall[1] - c*poiseuille_1d_pt_on_top_wall[2];
                        

                      Real rx = (i+0.5+ccent(i,j,k,0)) * dx[0];
                      Real ry = (j+0.5+ccent(i,j,k,1)) * dx[1];
                      Real rz = (k+0.5+ccent(i,j,k,2)) * dx[2];

                      auto dist = std::fabs(a*rx + b*ry + c*rz + d)/std::sqrt(a*a + b*b + c*c);

                      auto phi_mag = (!flag(i,j,k).isCovered()) ? dist * (H - dist) : 0.0;

                      Vector<Real> flow_norm(3, 0.0);

                      if(nfdir == 2) {
                         Real flow_norm_mag = std::sqrt(
                                                std::cos(alpha)*std::cos(alpha)*std::cos(gamma)*std::cos(gamma) 
                                                + std::sin(gamma)*std::sin(gamma));
                         flow_norm[0] = std::cos(alpha)*std::cos(gamma)/flow_norm_mag;
                         flow_norm[1] = std::sin(gamma)/flow_norm_mag;
                      }
                      else if(nfdir == 1) {
                         Real flow_norm_mag = std::sqrt(std::sin(alpha)*std::sin(alpha)
                                              + std::sin(gamma)*std::sin(gamma)); 
                         flow_norm[0] = -std::sin(alpha)/flow_norm_mag;
                         flow_norm[2] = std::sin(gamma)/flow_norm_mag;
                         
                      }
                      else if(nfdir == 0) {
                         Real flow_norm_mag = std::sqrt(
                                                std::cos(alpha)*std::cos(alpha)*std::cos(gamma)*std::cos(gamma) 
                                                + std::sin(alpha)*std::sin(alpha));
                         flow_norm[2] = std::cos(alpha)*std::cos(gamma)/flow_norm_mag;
                         flow_norm[1] = std::sin(alpha)/flow_norm_mag;
                      }
                      else {
                         AMREX_ALWAYS_ASSERT_WITH_MESSAGE(1==1, "Invalid flow direction");
                      }

                      fab(i,j,k,0) = phi_mag * flow_norm[0];
                      fab(i,j,k,1) = phi_mag * flow_norm[1];
                      fab(i,j,k,2) = phi_mag * flow_norm[2];

                      if( flag(i,j,k).isCovered()) {
                        fab_gx(i,j,k,0) = 0.0;
                        fab_gx(i,j,k,1) = 0.0;
                        fab_gx(i,j,k,2) = 0.0;
                        fab_gy(i,j,k,0) = 0.0;
                        fab_gy(i,j,k,1) = 0.0;
                        fab_gy(i,j,k,2) = 0.0;
                        fab_gz(i,j,k,0) = 0.0;
                        fab_gz(i,j,k,1) = 0.0;
                        fab_gz(i,j,k,2) = 0.0;
                      }
                      else {
                        Real rxl = i * dx[0];
                        Real ryl = (j+0.5+fcx(i,j,k,0)) * dx[1];
                        Real rzl = (k+0.5+fcx(i,j,k,1)) * dx[2];
                        Real fac = (H - 2*(a*rxl + b*ryl + c*rzl + d)/(std::sqrt(a*a + b*b + c*c)));
                        fab_gx(i,j,k,0) = (apx(i,j,k) == 0.0) ? 0.0 :
                           (a*flow_norm[0]/std::sqrt(a*a + b*b + c*c)) * fac * dx[0];
                        fab_gx(i,j,k,1) = (apx(i,j,k) == 0.0) ? 0.0 :
                           (a*flow_norm[1]/std::sqrt(a*a + b*b + c*c)) * fac * dx[0];
                        fab_gx(i,j,k,2) = (apx(i,j,k) == 0.0) ? 0.0 :
                           (a*flow_norm[2]/std::sqrt(a*a + b*b + c*c)) * fac * dx[0];

                        rxl = (i+0.5+fcy(i,j,k,0)) * dx[0];
                        ryl = j * dx[1];
                        rzl = (k+0.5+fcy(i,j,k,1)) * dx[2];
                        fac = (H - 2*(a*rxl + b*ryl + c*rzl + d)/(std::sqrt(a*a + b*b + c*c)));
                        fab_gy(i,j,k,0) = (apy(i,j,k) == 0.0) ? 0.0 :
                           (b*flow_norm[0]/std::sqrt(a*a + b*b + c*c)) * fac * dx[1];
                        fab_gy(i,j,k,1) = (apy(i,j,k) == 0.0) ? 0.0 :
                           (b*flow_norm[1]/std::sqrt(a*a + b*b + c*c)) * fac * dx[1];
                        fab_gy(i,j,k,2) = (apy(i,j,k) == 0.0) ? 0.0 :
                           (b*flow_norm[2]/std::sqrt(a*a + b*b + c*c)) * fac * dx[1];

                        rxl = (i+0.5+fcz(i,j,k,0)) * dx[0];
                        ryl = (j+0.5+fcz(i,j,k,1)) * dx[1];
                        rzl = k * dx[2];
                        fac = (H - 2*(a*rxl + b*ryl + c*rzl + d)/(std::sqrt(a*a + b*b + c*c)));
                        fab_gz(i,j,k,0) = (apz(i,j,k) == 0.0) ? 0.0 :
                           (c*flow_norm[0]/std::sqrt(a*a + b*b + c*c)) * fac * dx[2];
                        fab_gz(i,j,k,1) = (apz(i,j,k) == 0.0) ? 0.0 :
                           (c*flow_norm[1]/std::sqrt(a*a + b*b + c*c)) * fac * dx[2];
                        fab_gz(i,j,k,2) = (apz(i,j,k) == 0.0) ? 0.0 :
                           (c*flow_norm[2]/std::sqrt(a*a + b*b + c*c)) * fac * dx[2];
                      }

                      if(flag(i,j,k).isSingleValued()) {
                        Real rxeb = (i+0.5+bcent(i,j,k,0)) * dx[0];
                        Real ryeb = (j+0.5+bcent(i,j,k,1)) * dx[1];
                        Real rzeb = (k+0.5+bcent(i,j,k,2)) * dx[2];
                        Real fac = (H - 2*(a*rxeb+b*ryeb+c*rzeb+d)/(std::sqrt(a*a + b*b + c*c)));
                        Real dudx = (a*flow_norm[0]/std::sqrt(a*a + b*b + c*c)) * fac * dx[0];
                        Real dvdx = (a*flow_norm[1]/std::sqrt(a*a + b*b + c*c)) * fac * dx[0];
                        Real dwdx = (a*flow_norm[2]/std::sqrt(a*a + b*b + c*c)) * fac * dx[0];
                        Real dudy = (b*flow_norm[0]/std::sqrt(a*a + b*b + c*c)) * fac * dx[1];
                        Real dvdy = (b*flow_norm[1]/std::sqrt(a*a + b*b + c*c)) * fac * dx[1];
                        Real dwdy = (b*flow_norm[2]/std::sqrt(a*a + b*b + c*c)) * fac * dx[1];
                        Real dudz = (c*flow_norm[0]/std::sqrt(a*a + b*b + c*c)) * fac * dx[2];
                        Real dvdz = (c*flow_norm[1]/std::sqrt(a*a + b*b + c*c)) * fac * dx[2];
                        Real dwdz = (c*flow_norm[2]/std::sqrt(a*a + b*b + c*c)) * fac * dx[2];

                        fab_eb(i,j,k,0) = dudx*norm(i,j,k,0) + dudy*norm(i,j,k,1) + dudz*norm(i,j,k,2);
                        fab_eb(i,j,k,1) = dvdx*norm(i,j,k,0) + dvdy*norm(i,j,k,1) + dvdz*norm(i,j,k,2);
                        fab_eb(i,j,k,2) = dwdx*norm(i,j,k,0) + dwdy*norm(i,j,k,1) + dwdz*norm(i,j,k,2);
                      }

                   }
                   else { //3D grid-aligned
                      Real H = poiseuille_1d_height;
                      Real bot = poiseuille_1d_bottom;
                      int dir = poiseuille_1d_height_dir;
                      int fdir = poiseuille_1d_flow_dir;

                      fab(i,j,k,0) = 0.0;
                      fab(i,j,k,1) = 0.0;
                      fab(i,j,k,2) = 0.0;
                      fab_gx(i,j,k,0) = 0.0;
                      fab_gx(i,j,k,1) = 0.0;
                      fab_gx(i,j,k,2) = 0.0;
                      fab_gy(i,j,k,0) = 0.0;
                      fab_gy(i,j,k,1) = 0.0;
                      fab_gy(i,j,k,2) = 0.0;
                      fab_gz(i,j,k,0) = 0.0;
                      fab_gz(i,j,k,1) = 0.0;
                      fab_gz(i,j,k,2) = 0.0;

                      Real d = 0.0;
                      if(dir == 0) {
                        Real rx = (i+0.5+ccent(i,j,k,0)) * dx[0];
                        d = rx-bot;
                        fab(i,j,k,fdir) = (!flag(i,j,k).isCovered()) ? d * (H - d) : 0.0;

                        Real rxl = i * dx[0];
                        d = rxl-bot;
                        fab_gx(i,j,k,fdir) = (apx(i,j,k) == 0.0) ? 0.0 : (H - 2*d) * dx[0];

                        if(flag(i,j,k).isSingleValued()) {
                           fab_eb(i,j,k,0) = 0.0;
                           fab_eb(i,j,k,1) = 0.0;
                           fab_eb(i,j,k,2) = 0.0;

                           Real rxeb = (i+0.5+bcent(i,j,k,0)) * dx[0];
                           d = rxeb-bot;   
                           fab_eb(i,j,k,fdir) = (H - 2*d) * dx[0] * norm(i,j,k,0);
                        }

                      }
                      else if(dir == 1) {
                        Real ry = (j+0.5+ccent(i,j,k,1)) * dx[1];
                        d = ry-bot;
                        fab(i,j,k,fdir) = (!flag(i,j,k).isCovered()) ? d * (H - d) : 0.0;

                        Real ryl = j * dx[1];
                        d = ryl-bot;
                        fab_gy(i,j,k,fdir) = (apy(i,j,k) == 0.0) ? 0.0 : (H - 2*d) * dx[1];

                        if(flag(i,j,k).isSingleValued()) {
                           fab_eb(i,j,k,0) = 0.0;
                           fab_eb(i,j,k,1) = 0.0;
                           fab_eb(i,j,k,2) = 0.0;

                           Real ryeb = (j+0.5+bcent(i,j,k,1)) * dx[1];
                           d = ryeb-bot;   
                           fab_eb(i,j,k,fdir) = (H - 2*d) * dx[1] * norm(i,j,k,1);
                        }
                      }
                      else if(dir == 2) {
                        Real rz = (k+0.5+ccent(i,j,k,2)) * dx[2];
                        d = rz-bot;
                        fab(i,j,k,fdir) = (!flag(i,j,k).isCovered()) ? d * (H - d) : 0.0;

                        Real rzl = k * dx[2];
                        d = rzl-bot;
                        fab_gz(i,j,k,fdir) = (apz(i,j,k) == 0.0) ? 0.0 : (H - 2*d) * dx[2];

                        if(flag(i,j,k).isSingleValued()) {
                           fab_eb(i,j,k,0) = 0.0;
                           fab_eb(i,j,k,1) = 0.0;
                           fab_eb(i,j,k,2) = 0.0;

                           Real rzeb = (k+0.5+bcent(i,j,k,2)) * dx[2];
                           d = rzeb-bot;   
                           fab_eb(i,j,k,fdir) = (H - 2*d) * dx[2] * norm(i,j,k,2);
                        }
                      }
                      else {
                        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(1==1, "Invalid height direction");
                      }
                   }
#endif
               });
            }
            else {
               amrex::ParallelFor(bx,
               [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
               {
                   Real rx = (i+0.5)*dx[0];
                   Real ry = (j+0.5)*dx[1];
                   // fab(i,j,k) = std::sin(rx*2.*pi + 43.5)*std::sin(ry*2.*pi + 89.);
                   fab(i,j,k) = rx*(1.-rx)*ry*(1.-ry);
               });
            }
        }
    }
}

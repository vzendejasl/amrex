#include "MyTest.H"

#include <AMReX_MLEBABecLap.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_EB2.H>

#if (AMREX_SPACEDIM)== 2
#include <AMReX_EB_LeastSquares_2D_K.H>
#else
#include <AMReX_EB_LeastSquares_3D_K.H>
#endif

#include <cmath>

using namespace amrex;

MyTest::MyTest ()
{
    readParameters();

    initGrids();

    initializeEB();

    initData();
}

void
MyTest::compute_gradient ()
{
    int ilev = 0;

    bool is_eb_dirichlet = true;
    bool is_eb_inhomog  = false;

    int ncomp = phi[0].nComp();

    for (MFIter mfi(phi[ilev]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.fabbox();
        Array4<Real> const& phi_arr = phi[ilev].array(mfi);
        Array4<Real> const& phi_eb_arr = phieb[ilev].array(mfi);
        Array4<Real> const& grad_x_arr = grad_x[ilev].array(mfi);
        Array4<Real> const& grad_y_arr = grad_y[ilev].array(mfi);
        Array4<Real> const& grad_z_arr = grad_z[ilev].array(mfi);
        Array4<Real> const& grad_eb_arr = grad_eb[ilev].array(mfi);
        Array4<Real> const& ccentr_arr = ccentr[ilev].array(mfi);

        Array4<Real const> const& fcx   = (factory[ilev]->getFaceCent())[0]->const_array(mfi);
        Array4<Real const> const& fcy   = (factory[ilev]->getFaceCent())[1]->const_array(mfi);

        Array4<Real const> const& ccent = (factory[ilev]->getCentroid()).array(mfi);
        Array4<Real const> const& bcent = (factory[ilev]->getBndryCent()).array(mfi);
        Array4<Real const> const& apx   = (factory[ilev]->getAreaFrac())[0]->const_array(mfi);
        Array4<Real const> const& apy   = (factory[ilev]->getAreaFrac())[1]->const_array(mfi);
        Array4<Real const> const& norm  = (factory[ilev]->getBndryNormal()).array(mfi);

        const FabArray<EBCellFlagFab>* flags = &(factory[ilev]->getMultiEBCellFlagFab());
        Array4<EBCellFlag const> const& flag = flags->const_array(mfi);

        const auto dx = geom[ilev].CellSizeArray();

#if (AMREX_SPACEDIM > 2)
         Array4<Real const> const& fcz   = (factory[ilev]->getFaceCent())[2]->const_array(mfi);
         Array4<Real const> const& apz   = (factory[ilev]->getAreaFrac())[2]->const_array(mfi);
#endif

        amrex::ParallelFor(bx, ncomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            ccentr_arr(i,j,k,n) = ccent(i,j,k,n);
            Real yloc_on_xface = fcx(i,j,k,0);
            Real xloc_on_yface = fcy(i,j,k,0);
            Real nx = norm(i,j,k,0);
            Real ny = norm(i,j,k,1);

            // There is no need to set these to zero other than it makes using
            // amrvis a lot more friendly.
            if( flag(i,j,k).isCovered()){
              grad_x_arr(i,j,k,n)  = 0.0;
              grad_y_arr(i,j,k,n)  = 0.0;
              grad_z_arr(i,j,k,n)  = 0.0;

            }

#if (AMREX_SPACEDIM == 2)
            if( flag(i,j,k).isRegular() or flag(i,j,k).isSingleValued()){

              grad_x_arr(i,j,k,n) = (apx(i,j,k) == 0.0) ? 0.0 :
                grad_x_of_phi_on_centroids(i, j, k, n, phi_arr, phi_eb_arr,
                                           flag, ccent, bcent, apx, apy,
                                           yloc_on_xface, is_eb_dirichlet, is_eb_inhomog);


              grad_y_arr(i,j,k,n) = (apy(i,j,k) == 0.0) ? 0.0:
                grad_y_of_phi_on_centroids(i, j, k, n, phi_arr, phi_eb_arr,
                                           flag, ccent, bcent, apx, apy,
                                           xloc_on_yface, is_eb_dirichlet, is_eb_inhomog);


            }


            if (flag(i,j,k).isSingleValued())
              grad_eb_arr(i,j,k,n) = grad_eb_of_phi_on_centroids(i, j, k, n, phi_arr, phi_eb_arr,
                        flag, ccent, bcent, nx, ny, is_eb_inhomog);
#else

            Real zloc_on_xface = fcx(i,j,k,1);
            Real zloc_on_yface = fcy(i,j,k,1);
            Real xloc_on_zface = fcz(i,j,k,0);
            Real yloc_on_zface = fcz(i,j,k,1);

            Real nz = norm(i,j,k,2);

            if( flag(i,j,k).isRegular() or flag(i,j,k).isSingleValued()){

              grad_x_arr(i,j,k,n) = (apx(i,j,k) == 0.0) ? 0.0 :
                grad_x_of_phi_on_centroids(i, j, k, n, phi_arr, phi_eb_arr,
                                           flag, ccent, bcent, apx, apy, apz,
                                           yloc_on_xface, zloc_on_xface, is_eb_dirichlet, is_eb_inhomog);

              grad_y_arr(i,j,k,n) = (apy(i,j,k) == 0.0) ? 0.0:
                grad_y_of_phi_on_centroids(i, j, k, n, phi_arr, phi_eb_arr,
                                           flag, ccent, bcent, apx, apy, apz,
                                           xloc_on_yface, zloc_on_yface, is_eb_dirichlet, is_eb_inhomog);

              grad_z_arr(i,j,k,n) = (apz(i,j,k) == 0.0) ? 0.0:
                grad_z_of_phi_on_centroids(i, j, k, n, phi_arr, phi_eb_arr,
                                           flag, ccent, bcent, apx, apy, apz,
                                           xloc_on_zface, yloc_on_zface, is_eb_dirichlet, is_eb_inhomog);

            }


            if (flag(i,j,k).isSingleValued())
              grad_eb_arr(i,j,k,n) = grad_eb_of_phi_on_centroids(i, j, k, n, phi_arr, phi_eb_arr,
                        flag, ccent, bcent, nx, ny, nz, is_eb_inhomog);

#endif

        });
    }
}

void
MyTest::solve ()
{
    for (int ilev = 0; ilev <= max_level; ++ilev) {
        const MultiFab& vfrc = factory[ilev]->getVolFrac();
        MultiFab v(vfrc.boxArray(), vfrc.DistributionMap(), 1, 0,
                   MFInfo(), *factory[ilev]);
        MultiFab::Copy(v, vfrc, 0, 0, 1, 0);
        amrex::EB_set_covered(v, 1.0);
        amrex::Print() << "vfrc min = " << v.min(0) << std::endl;
    }

    std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_lobc;
    std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_hibc;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        if (geom[0].isPeriodic(idim)) {
            mlmg_lobc[idim] = LinOpBCType::Periodic;
            mlmg_hibc[idim] = LinOpBCType::Periodic;
        } else {
            mlmg_lobc[idim] = LinOpBCType::Dirichlet;
            mlmg_hibc[idim] = LinOpBCType::Dirichlet;
        }
    }

    LPInfo info;
    info.setMaxCoarseningLevel(max_coarsening_level);

    MLEBABecLap mleb (geom, grids, dmap, info, amrex::GetVecOfConstPtrs(factory));
    mleb.setMaxOrder(linop_maxorder);

    mleb.setDomainBC(mlmg_lobc, mlmg_hibc);

    for (int ilev = 0; ilev <= max_level; ++ilev) {
        mleb.setLevelBC(ilev, &phi[ilev]);
    }

    mleb.setScalars(scalars[0], scalars[1]);

    for (int ilev = 0; ilev <= max_level; ++ilev) {
        mleb.setACoeffs(ilev, acoef[ilev]);
        mleb.setBCoeffs(ilev, amrex::GetArrOfConstPtrs(bcoef[ilev]));
    }

    if (eb_is_dirichlet) {
        for (int ilev = 0; ilev <= max_level; ++ilev) {
            mleb.setEBDirichlet(ilev, phi[ilev], bcoef_eb[ilev]);
        }
    }

    MLMG mlmg(mleb);

    mlmg.apply(amrex::GetVecOfPtrs(rhs), amrex::GetVecOfPtrs(phi));
}

void
MyTest::writePlotfile ()
{
    Vector<MultiFab> plotmf(max_level+1);
    for (int ilev = 0; ilev <= max_level; ++ilev) {
        const MultiFab& vfrc = factory[ilev]->getVolFrac();

#if (AMREX_SPACEDIM == 2)
        plotmf[ilev].define(grids[ilev],dmap[ilev],14,0);
        MultiFab::Copy(plotmf[ilev], phi[ilev], 0, 0, 2, 0);
        MultiFab::Copy(plotmf[ilev], rhs[ilev], 0, 2, 1, 0);
        MultiFab::Copy(plotmf[ilev], vfrc, 0, 3, 1, 0);
        MultiFab::Copy(plotmf[ilev], phieb[ilev], 0, 4, 2, 0);
        MultiFab::Copy(plotmf[ilev], grad_x[ilev], 0, 6, 2, 0);
        MultiFab::Copy(plotmf[ilev], grad_y[ilev], 0, 8, 2, 0);
        MultiFab::Copy(plotmf[ilev], grad_eb[ilev], 0, 10, 2, 0);
        MultiFab::Copy(plotmf[ilev], ccentr[ilev], 0, 12, 2, 0);
    }
    WriteMultiLevelPlotfile(plot_file_name, max_level+1,
                            amrex::GetVecOfConstPtrs(plotmf),
                            {"u", "v", 
                             "rhs","vfrac", 
                             "ueb", "veb",
                             "dudx", "dvdx", 
                             "dudy", "dvdy", 
                             "dudn", "dvdn", 
                             "ccent_x", "ccent_y"},
                            geom, 0.0, Vector<int>(max_level+1,0),
                            Vector<IntVect>(max_level,IntVect{2}));
    
    Vector<MultiFab> plotmf_analytic(max_level+1);
    for (int ilev = 0; ilev <= max_level; ++ilev) {
        plotmf_analytic[ilev].define(grids[ilev],dmap[ilev],6,0);
        MultiFab::Copy(plotmf_analytic[ilev], grad_x_analytic[ilev], 0, 0, 2, 0);
        MultiFab::Copy(plotmf_analytic[ilev], grad_y_analytic[ilev], 0, 2, 2, 0);
        MultiFab::Copy(plotmf_analytic[ilev], grad_eb_analytic[ilev], 0, 4, 2, 0);
    }
    WriteMultiLevelPlotfile(plot_file_name + "-analytic", max_level+1,
                            amrex::GetVecOfConstPtrs(plotmf_analytic),
                            {"dudx", "dvdx", 
                             "dudy","dvdy",
                             "dudn","dvdn"},
                            geom, 0.0, Vector<int>(max_level+1,0),
                            Vector<IntVect>(max_level,IntVect{2}));
#else
        plotmf[ilev].define(grids[ilev],dmap[ilev],23,0);
        MultiFab::Copy(plotmf[ilev], phi[ilev], 0, 0, 3, 0);
        MultiFab::Copy(plotmf[ilev], rhs[ilev], 0, 3, 1, 0);
        MultiFab::Copy(plotmf[ilev], vfrc, 0, 4, 1, 0);
        MultiFab::Copy(plotmf[ilev], phieb[ilev], 0, 5, 3, 0);
        MultiFab::Copy(plotmf[ilev], grad_x[ilev], 0, 8, 3, 0);
        MultiFab::Copy(plotmf[ilev], grad_y[ilev], 0, 11, 3, 0);
        MultiFab::Copy(plotmf[ilev], grad_z[ilev], 0, 14, 3, 0);
        MultiFab::Copy(plotmf[ilev], grad_eb[ilev], 0, 17, 3, 0);
        MultiFab::Copy(plotmf[ilev], ccentr[ilev], 0, 20, 3, 0);
    }
    WriteMultiLevelPlotfile(plot_file_name, max_level+1,
                            amrex::GetVecOfConstPtrs(plotmf),
                            {"u", "v", "w",
                             "rhs","vfrac", 
                             "ueb", "veb", "web",
                             "dudx", "dvdx", "dwdx",
                             "dudy", "dvdy", "dwdy",
                             "dudz", "dvdz", "dwdz",
                             "dudn", "dvdn", "dwdn",
                             "ccent_x", "ccent_y", "ccent_z"},
                            geom, 0.0, Vector<int>(max_level+1,0),
                            Vector<IntVect>(max_level,IntVect{2}));
    
    Vector<MultiFab> plotmf_analytic(max_level+1);
    for (int ilev = 0; ilev <= max_level; ++ilev) {
        plotmf_analytic[ilev].define(grids[ilev],dmap[ilev],12,0);
        MultiFab::Copy(plotmf_analytic[ilev], grad_x_analytic[ilev], 0, 0, 3, 0);
        MultiFab::Copy(plotmf_analytic[ilev], grad_y_analytic[ilev], 0, 3, 3, 0);
        MultiFab::Copy(plotmf_analytic[ilev], grad_z_analytic[ilev], 0, 6, 3, 0);
        MultiFab::Copy(plotmf_analytic[ilev], grad_eb_analytic[ilev], 0, 9, 3, 0);
    }
    WriteMultiLevelPlotfile(plot_file_name + "-analytic", max_level+1,
                            amrex::GetVecOfConstPtrs(plotmf_analytic),
                            {"dudx", "dvdx","dwdx",
                             "dudy","dvdy","dwdy",
                             "dudz","dvdz","dwdz",
                             "dudn","dvdn","dwdn"},
                            geom, 0.0, Vector<int>(max_level+1,0),
                            Vector<IntVect>(max_level,IntVect{2}));
    
#endif

}

void
MyTest::readParameters ()
{
    ParmParse pp;
    pp.query("max_level", max_level);
    pp.query("n_cell", n_cell);
    pp.query("max_grid_size", max_grid_size);

    pp.queryarr("is_periodic", is_periodic);

    pp.query("eb_is_dirichlet", eb_is_dirichlet);

    pp.query("plot_file", plot_file_name);

    pp.queryarr("prob_lo", prob_lo);
    pp.queryarr("prob_hi", prob_hi);

    scalars.resize(2);
    if (is_periodic[0]) {
        scalars[0] = 0.0;
        scalars[1] = 1.0;
    }
    else if (is_periodic[1]) {
        scalars[0] = 1.0;
        scalars[1] = 0.0;
    }
    else {
        scalars[0] = 1.0;
        scalars[1] = 1.0;
    }
    pp.queryarr("scalars", scalars);

    pp.query("verbose", verbose);
    pp.query("bottom_verbose", bottom_verbose);
    pp.query("max_iter", max_iter);
    pp.query("max_fmg_iter", max_fmg_iter);
    pp.query("max_bottom_iter", max_bottom_iter);
    pp.query("bottom_reltol", bottom_reltol);
    pp.query("reltol", reltol);
    pp.query("linop_maxorder", linop_maxorder);
    pp.query("max_coarsening_level", max_coarsening_level);
#ifdef AMREX_USE_HYPRE
    pp.query("use_hypre", use_hypre);
#endif
#ifdef AMREX_USE_PETSC
    pp.query("use_petsc",use_petsc);
#endif
    pp.query("use_poiseuille_1d", use_poiseuille_1d);
    pp.query("poiseuille_1d_askew", poiseuille_1d_askew);
    pp.queryarr("poiseuille_1d_pt_on_top_wall",poiseuille_1d_pt_on_top_wall);
    pp.query("poiseuille_1d_height",poiseuille_1d_height);
    pp.query("poiseuille_1d_rotation",poiseuille_1d_rotation);
    pp.queryarr("poiseuille_1d_askew_rotation",poiseuille_1d_askew_rotation);
    pp.query("poiseuille_1d_flow_dir", poiseuille_1d_flow_dir);
    pp.query("poiseuille_1d_height_dir", poiseuille_1d_height_dir);
    pp.query("poiseuille_1d_bottom", poiseuille_1d_bottom);
    pp.query("poiseuille_1d_no_flow_dir", poiseuille_1d_no_flow_dir);
}

void
MyTest::initGrids ()
{
    int nlevels = max_level + 1;
    geom.resize(nlevels);
    grids.resize(nlevels);

    RealBox rb({AMREX_D_DECL(prob_lo[0],prob_lo[1],prob_lo[2])}, {AMREX_D_DECL(prob_hi[0],prob_hi[1],prob_hi[2])});
    std::array<int,AMREX_SPACEDIM> isperiodic{AMREX_D_DECL(is_periodic[0],is_periodic[1],is_periodic[2])};
    Geometry::Setup(&rb, 0, isperiodic.data());
    Box domain0(IntVect{AMREX_D_DECL(0,0,0)}, IntVect{AMREX_D_DECL(n_cell-1,n_cell-1,n_cell-1)});
    Box domain = domain0;
    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        geom[ilev].define(domain);
        domain.refine(ref_ratio);
    }

    domain = domain0;
    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        grids[ilev].define(domain);
        grids[ilev].maxSize(max_grid_size);
        domain.grow(-n_cell/4);   // fine level cover the middle of the coarse domain
        domain.refine(ref_ratio);
    }
}

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
        ccentr[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, 1, MFInfo(), *factory[ilev]);
        rhs[ilev].define(grids[ilev], dmap[ilev], 1, 0, MFInfo(), *factory[ilev]);
        acoef[ilev].define(grids[ilev], dmap[ilev], 1, 0, MFInfo(), *factory[ilev]);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            bcoef[ilev][idim].define(amrex::convert(grids[ilev],IntVect::TheDimensionVector(idim)),
                                     dmap[ilev], 1, 0, MFInfo(), *factory[ilev]);
        }
        if (eb_is_dirichlet) {
            bcoef_eb[ilev].define(grids[ilev], dmap[ilev], 1, 0, MFInfo(), *factory[ilev]);
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
        ccentr[ilev].setVal(0.0);
        rhs[ilev].setVal(0.0);
        acoef[ilev].setVal(1.0);
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
                   if(poiseuille_1d_askew) {
                      Real H = poiseuille_1d_height;
                      int nfdir = poiseuille_1d_no_flow_dir;
                      Real alpha = (poiseuille_1d_askew_rotation[0]/180.)*M_PI;
                      Real gamma = (poiseuille_1d_askew_rotation[1]/180.)*M_PI;

                      Real a = std::sin(gamma);
                      Real b = -std::cos(alpha)*std::cos(gamma);
                      Real c = std::sin(alpha);
                      Real d = -a*poiseuille_1d_pt_on_top_wall[0] - b*poiseuille_1d_pt_on_top_wall[1] - c*poiseuille_1d_pt_on_top_wall[2];
                        

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
                        fab_gx(i,j,k,0) = (apx(i,j,k) == 0.0) ? 0.0 : (a*flow_norm[0]/std::sqrt(a*a + b*b + c*c)) * fac * dx[0];
                        fab_gx(i,j,k,1) = (apx(i,j,k) == 0.0) ? 0.0 : (a*flow_norm[1]/std::sqrt(a*a + b*b + c*c)) * fac * dx[0];
                        fab_gx(i,j,k,2) = (apx(i,j,k) == 0.0) ? 0.0 : (a*flow_norm[2]/std::sqrt(a*a + b*b + c*c)) * fac * dx[0];

                        rxl = (i+0.5+fcy(i,j,k,0)) * dx[0];
                        ryl = j * dx[1];
                        rzl = (k+0.5+fcy(i,j,k,1)) * dx[2];
                        fac = (H - 2*(a*rxl + b*ryl + c*rzl + d)/(std::sqrt(a*a + b*b + c*c)));
                        fab_gy(i,j,k,0) = (apy(i,j,k) == 0.0) ? 0.0 : (b*flow_norm[0]/std::sqrt(a*a + b*b + c*c)) * fac * dx[1];
                        fab_gy(i,j,k,1) = (apy(i,j,k) == 0.0) ? 0.0 : (b*flow_norm[1]/std::sqrt(a*a + b*b + c*c)) * fac * dx[1];
                        fab_gy(i,j,k,2) = (apy(i,j,k) == 0.0) ? 0.0 : (b*flow_norm[2]/std::sqrt(a*a + b*b + c*c)) * fac * dx[1];

                        rxl = (i+0.5+fcz(i,j,k,0)) * dx[0];
                        ryl = (j+0.5+fcz(i,j,k,1)) * dx[1];
                        rzl = k * dx[2];
                        fac = (H - 2*(a*rxl + b*ryl + c*rzl + d)/(std::sqrt(a*a + b*b + c*c)));
                        fab_gz(i,j,k,0) = (apz(i,j,k) == 0.0) ? 0.0 : (c*flow_norm[0]/std::sqrt(a*a + b*b + c*c)) * fac * dx[2];
                        fab_gz(i,j,k,1) = (apz(i,j,k) == 0.0) ? 0.0 : (c*flow_norm[1]/std::sqrt(a*a + b*b + c*c)) * fac * dx[2];
                        fab_gz(i,j,k,2) = (apz(i,j,k) == 0.0) ? 0.0 : (c*flow_norm[2]/std::sqrt(a*a + b*b + c*c)) * fac * dx[2];
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
                   else { //grid-aligned
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

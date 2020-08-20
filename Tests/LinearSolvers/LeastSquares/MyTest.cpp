#include "MyTest.H"

#include <AMReX_MLEBABecLap.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_EB2.H>

#include <AMReX_EB_LeastSquares_2D_K.H>

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
        Array4<Real> const& grad_eb_arr = grad_eb[ilev].array(mfi);
        Array4<Real> const& ccent_x_arr = ccent_x[ilev].array(mfi);
        Array4<Real> const& ccent_y_arr = ccent_y[ilev].array(mfi);

        Array4<Real const> const& fcx   = (factory[ilev]->getFaceCent())[0]->const_array(mfi);
        Array4<Real const> const& fcy   = (factory[ilev]->getFaceCent())[1]->const_array(mfi);

        Array4<Real const> const& ccent = (factory[ilev]->getCentroid()).array(mfi);
        Array4<Real const> const& bcent = (factory[ilev]->getBndryCent()).array(mfi);
        Array4<Real const> const& apx   = (factory[ilev]->getAreaFrac())[0]->const_array(mfi);
        Array4<Real const> const& apy   = (factory[ilev]->getAreaFrac())[1]->const_array(mfi);
        Array4<Real const> const& norm  = (factory[ilev]->getBndryNormal()).array(mfi);

        const FabArray<EBCellFlagFab>* flags = &(factory[ilev]->getMultiEBCellFlagFab());
        Array4<EBCellFlag const> const& flag = flags->const_array(mfi);

        amrex::ParallelFor(bx, ncomp, 
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Real yloc_on_xface = fcx(i,j,k);
            Real xloc_on_yface = fcy(i,j,k);
            Real nx = norm(i,j,k,0);
            Real ny = norm(i,j,k,1);
            ccent_x_arr(i,j,k) = ccent(i,j,k,0);
            ccent_y_arr(i,j,k) = ccent(i,j,k,1);

            grad_x_arr(i,j,k) = 
               grad_x_of_phi_on_centroids(i, j, k, n, 
                                          phi_arr,
                                          phi_eb_arr,
                                          flag,
                                          ccent, bcent, 
                                          apx, apy, 
                                          yloc_on_xface,
                                          is_eb_dirichlet, is_eb_inhomog);
            grad_y_arr(i,j,k) = 
               grad_y_of_phi_on_centroids(i, j, k, n, 
                                          phi_arr,
                                          phi_eb_arr,
                                          flag,
                                          ccent, bcent, 
                                          apx, apy, 
                                          xloc_on_yface,
                                          is_eb_dirichlet, is_eb_inhomog);
            grad_eb_arr(i,j,k) = 
               grad_eb_of_phi_on_centroids(i, j, k, n, 
                                          phi_arr,
                                          phi_eb_arr,
                                          flag,
                                          ccent, bcent, 
                                          nx, ny, 
                                          is_eb_inhomog);
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
        plotmf[ilev].define(grids[ilev],dmap[ilev],9,0);
        MultiFab::Copy(plotmf[ilev], phi[ilev], 0, 0, 1, 0);
        MultiFab::Copy(plotmf[ilev], rhs[ilev], 0, 1, 1, 0);
        MultiFab::Copy(plotmf[ilev], vfrc, 0, 2, 1, 0);    
        MultiFab::Copy(plotmf[ilev], phieb[ilev], 0, 3, 1, 0);
        MultiFab::Copy(plotmf[ilev], grad_x[ilev], 0, 4, 1, 0);
        MultiFab::Copy(plotmf[ilev], grad_y[ilev], 0, 5, 1, 0);
        MultiFab::Copy(plotmf[ilev], grad_eb[ilev], 0, 6, 1, 0);
        MultiFab::Copy(plotmf[ilev], ccent_x[ilev], 0, 7, 1, 0);
        MultiFab::Copy(plotmf[ilev], ccent_y[ilev], 0, 8, 1, 0);
    }
    WriteMultiLevelPlotfile(plot_file_name, max_level+1,
                            amrex::GetVecOfConstPtrs(plotmf),
                            {"phi","rhs","vfrac", "phieb", "grad_x", "grad_y", "grad_eb", "ccent_x", "ccent_y"},
                            geom, 0.0, Vector<int>(max_level+1,0),
                            Vector<IntVect>(max_level,IntVect{2}));
                            
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
    } else {
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
    pp.query("poiseuille_1d_left_wall",poiseuille_1d_left_wall);
    pp.query("poiseuille_1d_right_wall",poiseuille_1d_right_wall);
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
    grad_y.resize(nlevels);
    grad_eb.resize(nlevels);
    ccent_x.resize(nlevels);
    ccent_y.resize(nlevels);
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

        phi[ilev].define(grids[ilev], dmap[ilev], 1, 1, MFInfo(), *factory[ilev]);
        phieb[ilev].define(grids[ilev], dmap[ilev], 1, 1, MFInfo(), *factory[ilev]);
        grad_x[ilev].define(grids[ilev], dmap[ilev], 1, 1, MFInfo(), *factory[ilev]);
        grad_y[ilev].define(grids[ilev], dmap[ilev], 1, 1, MFInfo(), *factory[ilev]);
        grad_eb[ilev].define(grids[ilev], dmap[ilev], 1, 1, MFInfo(), *factory[ilev]);
        ccent_x[ilev].define(grids[ilev], dmap[ilev], 1, 1, MFInfo(), *factory[ilev]);
        ccent_y[ilev].define(grids[ilev], dmap[ilev], 1, 1, MFInfo(), *factory[ilev]);
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
        grad_y[ilev].setVal(1e40);
        grad_eb[ilev].setVal(1e40);
        ccent_x[ilev].setVal(0.0);
        ccent_y[ilev].setVal(0.0);
        rhs[ilev].setVal(0.0);
        acoef[ilev].setVal(1.0);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            bcoef[ilev][idim].setVal(1.0);
        }

        const auto dx = geom[ilev].CellSizeArray();

        const Real pi = 4.0*std::atan(1.0);

        for (MFIter mfi(rhs[ilev]); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.fabbox();
            Array4<Real> const& fab = phi[ilev].array(mfi);
            if (use_poiseuille_1d) {
               Array4<Real const> const& fcy   = (factory[ilev]->getFaceCent())[1]->const_array(mfi);
               amrex::ParallelFor(bx,
               [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
               {
                   auto H = poiseuille_1d_right_wall - poiseuille_1d_left_wall;
                   auto lw = poiseuille_1d_left_wall;
                   Real rx = (i+0.5 + fcy(i,j,k))*dx[0];
                   fab(i,j,k) = (rx - lw) * (1. - (rx - lw));
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


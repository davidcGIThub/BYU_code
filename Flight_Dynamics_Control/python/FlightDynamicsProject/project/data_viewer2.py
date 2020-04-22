from state_plotter.Plotter import Plotter
from state_plotter.plotter_args import *


class data_viewer2:
    def __init__(self):
        time_window_length=100
        self.plotter = Plotter(plotting_frequency=100, # refresh plot every 100 time steps
                               time_window=time_window_length)  # plot last time_window seconds of data
        # set up the plot window
        # define first row
        pn_plots = PlotboxArgs(plots=['pn', 'pn_EKF' ,'pn_FSD_EKF','pn_FSI_EKF' ],
                               labels={'left': 'pn(m)'},
                               time_window=time_window_length)
        pe_plots = PlotboxArgs(plots=['pe', 'pe_EKF' , 'pe_FSD_EKF', 'pe_FSI_EKF'],
                               labels={'left': 'pe(m)'},
                               time_window=time_window_length)
        h_plots = PlotboxArgs(plots=['h', 'h_EKF', 'h_FSD_EKF', 'h_FSI_EKF'],
                              labels={'left': 'h(m)'},
                              time_window=time_window_length)
        first_row = [pn_plots, pe_plots, h_plots]

        # define second row
        Va_plots = PlotboxArgs(plots=['Va', 'Va_EKF', 'Va_FSD_EKF', 'Va_FSI_EKF'],
                               labels={'left': 'Va(m/s)'},
                               time_window=time_window_length)
        Vg_plots = PlotboxArgs(plots=['Vg', 'Vg_EKF' , 'Vg_FSD_EKF', 'Vg_FSI_EKF'],
                               labels={'left': 'Vg(m/s)'},
                               time_window=time_window_length)
        chi_plots = PlotboxArgs(plots=['chi', 'chi_EKF', 'chi_FSD_EKF', 'chi_FSI_EKF'],
                                labels={'left': 'chi(deg)'},
                                rad2deg=True,
                                time_window=time_window_length)
        second_row = [Va_plots, Vg_plots, chi_plots]

        # define third row
        phi_plots = PlotboxArgs(plots=['phi', 'phi_EKF', 'phi_FSD_EKF', 'phi_FSI_EKF'],
                                labels={'left': 'phi(deg)', 'bottom': 'Time (s)'},
                                rad2deg=True,
                                time_window=time_window_length)
        theta_plots = PlotboxArgs(plots=['theta', 'theta_EKF', 'theta_FSD_EKF', 'theta_FSI_EKF'],
                                  labels={'left': 'theta(deg)', 'bottom': 'Time (s)'},
                                  rad2deg=True,
                                  time_window=time_window_length)
        psi_plots = PlotboxArgs(plots=['psi', 'psi_EKF' , 'psi_FSD_EKF', 'psi_FSI_EKF'],
                                labels={'left': 'psi(deg)', 'bottom': 'Time (s)'},
                                rad2deg=True,
                                time_window=time_window_length)
        third_row = [phi_plots, theta_plots, psi_plots]
        plots = [first_row,
                 second_row,
                 third_row
                 ]
        # Add plots to the window
        self.plotter.add_plotboxes(plots)
        # Define and label vectors for more convenient/natural data input
        self.plotter.define_input_vector('true_state', ['pn', 'pe', 'h', 'Va', 'alpha', 'beta', 'phi', 'theta', 'chi',
                                                        'p', 'q', 'r', 'Vg', 'wn', 'we', 'psi', 'bx', 'by', 'bz'])
        self.plotter.define_input_vector('EKF', ['pn_EKF', 'pe_EKF', 'h_EKF', 'Va_EKF', 'alpha_EKF', 'beta_EKF',
                                                             'phi_EKF', 'theta_EKF', 'chi_EKF', 'p_EKF', 'q_EKF', 'r_EKF',
                                                             'Vg_EKF', 'wn_EKF', 'we_EKF', 'psi_EKF', 'bx_EKF', 'by_EKF', 'bz_EKF'])
        self.plotter.define_input_vector('FSD_EKF', ['pn_FSD_EKF', 'pe_FSD_EKF', 'h_FSD_EKF', 'Va_FSD_EKF', 'alpha_FSD_EKF', 'beta_FSD_EKF',
                                                             'phi_FSD_EKF', 'theta_FSD_EKF', 'chi_FSD_EKF', 'p_FSD_EKF', 'q_FSD_EKF', 'r_FSD_EKF',
                                                             'Vg_FSD_EKF', 'wn_FSD_EKF', 'we_FSD_EKF', 'psi_FSD_EKF2', 'bx_FSD_EKF', 'by_FSD_EKF', 'bz_FSD_EKF'])
        self.plotter.define_input_vector('FSI_EKF', ['pn_FSI_EKF', 'pe_FSI_EKF', 'h_FSI_EKF', 'Va_FSI_EKF', 'alpha_FSI_EKF', 'beta_FSI_EKF',
                                                             'phi_FSI_EKF', 'theta_FSI_EKF', 'chi_FSI_EKF', 'p_FSI_EKF', 'q_FSI_EKF', 'r_FSI_EKF',
                                                             'Vg_FSI_EKF', 'wn_FSI_EKF', 'we_FSI_EKF', 'psi_FSI_EKF2', 'bx_FSI_EKF', 'by_FSI_EKF', 'bz_FSI_EKF'])
        # plot timer
        self.time = 0.

    def update(self, true_state, EKF, FSD_EKF,FSI_EKF, ts):
        true_state_list = [true_state.pn, true_state.pe, true_state.h,
                           true_state.Va, true_state.alpha, true_state.beta,
                           true_state.phi, true_state.theta, true_state.chi,
                           true_state.p, true_state.q, true_state.r,
                           true_state.Vg, true_state.wn, true_state.we, true_state.psi,
                           true_state.bx, true_state.by, true_state.bz]
        EKF_list = [EKF.pn, EKF.pe, EKF.h,
                                EKF.Va, EKF.alpha, EKF.beta,
                                EKF.phi, EKF.theta, EKF.chi,
                                EKF.p, EKF.q, EKF.r,
                                EKF.Vg, EKF.wn, EKF.we, EKF.psi,
                                EKF.bx, EKF.by, EKF.bz]
        FSD_EKF_list = [FSD_EKF.pn, FSD_EKF.pe, FSD_EKF.h,
                                FSD_EKF.Va, FSD_EKF.alpha, FSD_EKF.beta,
                                FSD_EKF.phi, FSD_EKF.theta, FSD_EKF.chi,
                                FSD_EKF.p, FSD_EKF.q, FSD_EKF.r,
                                FSD_EKF.Vg, FSD_EKF.wn, FSD_EKF.we, FSD_EKF.psi,
                                FSD_EKF.bx, FSD_EKF.by, FSD_EKF.bz]
        FSI_EKF_list = [FSI_EKF.pn, FSI_EKF.pe, FSI_EKF.h,
                                FSI_EKF.Va, FSI_EKF.alpha, FSI_EKF.beta,
                                FSI_EKF.phi, FSI_EKF.theta, FSI_EKF.chi,
                                FSI_EKF.p, FSI_EKF.q, FSI_EKF.r,
                                FSI_EKF.Vg, FSI_EKF.wn, FSI_EKF.we, FSI_EKF.psi,
                                FSI_EKF.bx, FSI_EKF.by, FSI_EKF.bz]
        self.plotter.add_vector_measurement('true_state', true_state_list, self.time)
        self.plotter.add_vector_measurement('EKF', EKF_list, self.time)
        self.plotter.add_vector_measurement('FSD_EKF', FSD_EKF_list, self.time)
        self.plotter.add_vector_measurement('FSI_EKF', FSI_EKF_list, self.time)

        # Update and display the plot
        self.plotter.update_plots()

        # increment time
        self.time += ts




from state_plotter.Plotter import Plotter
from state_plotter.plotter_args import *


class data_viewer2:
    def __init__(self):
        time_window_length=100
        self.plotter = Plotter(plotting_frequency=100, # refresh plot every 100 time steps
                               time_window=time_window_length)  # plot last time_window seconds of data
        # set up the plot window
        # define first row
        pn_plots = PlotboxArgs(plots=['pn', 'pn_e' ,'pn_e2' ],
                               labels={'left': 'pn(m)', 'bottom': 'Time (s)'},
                               time_window=time_window_length)
        pe_plots = PlotboxArgs(plots=['pe', 'pe_e' , 'pe_e2'],
                               labels={'left': 'pe(m)', 'bottom': 'Time (s)'},
                               time_window=time_window_length)
        h_plots = PlotboxArgs(plots=['h', 'h_e', 'h_e2'],
                              labels={'left': 'h(m)', 'bottom': 'Time (s)'},
                              time_window=time_window_length)
        wind_plots = PlotboxArgs(plots=['wn', 'wn_e', 'wn_e2' , 'we', 'we_e' ,'we_e2'],
                                 labels={'left': 'wind(m/s)', 'bottom': 'Time (s)'},
                                 time_window=time_window_length)
        first_row = [pn_plots, pe_plots, h_plots, wind_plots]

        # define second row
        Va_plots = PlotboxArgs(plots=['Va', 'Va_e', 'Va_e2'],
                               labels={'left': 'Va(m/s)', 'bottom': 'Time (s)'},
                               time_window=time_window_length)
        alpha_plots = PlotboxArgs(plots=['alpha', 'alpha_e' , 'alpha_e2'],
                                  labels={'left': 'alpha(deg)', 'bottom': 'Time (s)'},
                                  rad2deg=True,
                                  time_window=time_window_length)
        beta_plots = PlotboxArgs(plots=['beta', 'beta_e' , 'beta_e2'],
                                 labels={'left': 'beta(deg)', 'bottom': 'Time (s)'},
                                 rad2deg=True,
                                 time_window=time_window_length)
        Vg_plots = PlotboxArgs(plots=['Vg', 'Vg_e' , 'Vg_e2'],
                               labels={'left': 'Vg(m/s)', 'bottom': 'Time (s)'},
                               time_window=time_window_length)
        second_row = [Va_plots, alpha_plots, beta_plots, Vg_plots]

        # define third row
        phi_plots = PlotboxArgs(plots=['phi', 'phi_e', 'phi_e2'],
                                labels={'left': 'phi(deg)', 'bottom': 'Time (s)'},
                                rad2deg=True,
                                time_window=time_window_length)
        theta_plots = PlotboxArgs(plots=['theta', 'theta_e', 'theta_e2'],
                                  labels={'left': 'theta(deg)', 'bottom': 'Time (s)'},
                                  rad2deg=True,
                                  time_window=time_window_length)
        psi_plots = PlotboxArgs(plots=['psi', 'psi_e' , 'psi_e2'],
                                labels={'left': 'psi(deg)', 'bottom': 'Time (s)'},
                                rad2deg=True,
                                time_window=time_window_length)
        chi_plots = PlotboxArgs(plots=['chi', 'chi_e', 'chi_e2'],
                                labels={'left': 'chi(deg)', 'bottom': 'Time (s)'},
                                rad2deg=True,
                                time_window=time_window_length)
        third_row = [phi_plots, theta_plots, psi_plots, chi_plots]

        # define fourth row
        p_plots = PlotboxArgs(plots=['p', 'p_e' , 'p_e2'],
                              labels={'left': 'p(deg/s)', 'bottom': 'Time (s)'},
                              rad2deg=True,
                              time_window=time_window_length)
        q_plots = PlotboxArgs(plots=['q', 'q_e', 'q_e2'],
                              labels={'left': 'q(deg/s)', 'bottom': 'Time (s)'},
                              rad2deg=True,
                              time_window=time_window_length)
        r_plots = PlotboxArgs(plots=['r', 'r_e', 'r_e2'],
                              labels={'left': 'r(deg)', 'bottom': 'Time (s)'},
                              rad2deg=True,
                              time_window=time_window_length)
        gyro_plots = PlotboxArgs(plots=['bx', 'bx_e', 'bx_e2' , 'by', 'by_e','by_e2' , 'bz', 'bz_e' , 'bz_e2'],
                                 labels={'left': 'bias(deg/s)', 'bottom': 'Time (s)'},
                                 rad2deg=True,
                                 time_window=time_window_length)
        fourth_row = [p_plots, q_plots, r_plots, gyro_plots]
        plots = [first_row,
                 second_row,
                 third_row,
                 fourth_row
                 ]
        # Add plots to the window
        self.plotter.add_plotboxes(plots)
        # Define and label vectors for more convenient/natural data input
        self.plotter.define_input_vector('true_state', ['pn', 'pe', 'h', 'Va', 'alpha', 'beta', 'phi', 'theta', 'chi',
                                                        'p', 'q', 'r', 'Vg', 'wn', 'we', 'psi', 'bx', 'by', 'bz'])
        self.plotter.define_input_vector('estimated_state', ['pn_e', 'pe_e', 'h_e', 'Va_e', 'alpha_e', 'beta_e',
                                                             'phi_e', 'theta_e', 'chi_e', 'p_e', 'q_e', 'r_e',
                                                             'Vg_e', 'wn_e', 'we_e', 'psi_e', 'bx_e', 'by_e', 'bz_e'])
        self.plotter.define_input_vector('estimated_state2', ['pn_e2', 'pe_e2', 'h_e2', 'Va_e2', 'alpha_e2', 'beta_e2',
                                                             'phi_e2', 'theta_e2', 'chi_e2', 'p_e2', 'q_e2', 'r_e2',
                                                             'Vg_e2', 'wn_e2', 'we_e2', 'psi_e2', 'bx_e2', 'by_e2', 'bz_e2'])
        # plot timer
        self.time = 0.

    def update(self, true_state, estimated_state, estimated_state2, ts):
        estimated_state_list = [estimated_state.pn, estimated_state.pe, estimated_state.h,
                                estimated_state.Va, estimated_state.alpha, estimated_state.beta,
                                estimated_state.phi, estimated_state.theta, estimated_state.chi,
                                estimated_state.p, estimated_state.q, estimated_state.r,
                                estimated_state.Vg, estimated_state.wn, estimated_state.we, estimated_state.psi,
                                estimated_state.bx, estimated_state.by, estimated_state.bz]
        true_state_list = [true_state.pn, true_state.pe, true_state.h,
                           true_state.Va, true_state.alpha, true_state.beta,
                           true_state.phi, true_state.theta, true_state.chi,
                           true_state.p, true_state.q, true_state.r,
                           true_state.Vg, true_state.wn, true_state.we, true_state.psi,
                           true_state.bx, true_state.by, true_state.bz]
        estimated_state_list2 = [estimated_state2.pn, estimated_state2.pe, estimated_state2.h,
                                estimated_state2.Va, estimated_state2.alpha, estimated_state2.beta,
                                estimated_state2.phi, estimated_state2.theta, estimated_state2.chi,
                                estimated_state2.p, estimated_state2.q, estimated_state2.r,
                                estimated_state2.Vg, estimated_state2.wn, estimated_state2.we, estimated_state2.psi,
                                estimated_state2.bx, estimated_state2.by, estimated_state2.bz]
        self.plotter.add_vector_measurement('true_state', true_state_list, self.time)
        self.plotter.add_vector_measurement('estimated_state', estimated_state_list, self.time)
        self.plotter.add_vector_measurement('estimated_state2', estimated_state_list2, self.time)

        # Update and display the plot
        self.plotter.update_plots()

        # increment time
        self.time += ts




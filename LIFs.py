#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""LIF"""
###############################################################################
###############################################################################
import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
from neurodynex3.leaky_integrate_and_fire import LIF
import neurodynex3.exponential_integrate_fire.exp_IF as exp_IF
from neurodynex3.adex_model import AdEx
from neurodynex3.tools import plot_tools

### 1.LIF neuron
'''
注：A LIF neuron is determined by the following parameters: Resting potential, reset voltage, 
   firing threshold, membrane resistance, membrane time-scale, absolute refractory period. 
   By injecting a known test current into a LIF neuron (with unknown parameters), you can 
   determine the neuron properties from the voltage response.
'''
input_current = b2.TimedArray(5*np.random.rand(200,1)*b2.namp, dt=1.0*b2.ms)
state_monitor, spike_monitor = LIF.simulate_LIF_neuron(input_current,
                                                       simulation_time = 100 * b2.ms,
                                                       firing_threshold = -50 * b2.mV,
                                                       membrane_resistance = 10 * b2.Mohm,
                                                       membrane_time_scale = 8 * b2.ms,
                                                       abs_refractory_period = 2.0 * b2.ms)
plot_tools.plot_voltage_and_current_traces(state_monitor, 
                                           input_current,
                                           title="input_current", 
                                           firing_threshold=-50 * b2.mV)
print("number of spikes: {}".format(len(spike_monitor.t)))



### 2.Exponential Integrate-and-Fire model
input_current = b2.TimedArray(np.random.rand(200,1)*b2.namp, dt=1.0*b2.ms)
state_monitor, spike_monitor = exp_IF.simulate_exponential_IF_neuron(I_stim=input_current, 
                                                                     v_rheobase=-55 * b2.mV,
                                                                     simulation_time=200*b2.ms)
plot_tools.plot_voltage_and_current_traces(state_monitor, 
                                           input_current,
                                           title="input_current",
                                           firing_threshold=-55 * b2.mV)
print("nr of spikes: {}".format(spike_monitor.count[0]))



### 3.AdEx(Adaptive Exponential Integrate-and-Fire model()
# Tonic
input_current = LIF.input_factory.get_step_current(10, 250, 1. * b2.ms, 65.0 * b2.pA)
state_monitor, spike_monitor = AdEx.simulate_AdEx_neuron(tau_m=20 * b2.ms,
                                                         tau_w=30 * b2.ms,
                                                         a=0 * b2.nS,
                                                         b=60 * b2.pA,
                                                         v_reset=-55 * b2.mV,
                                                         I_stim=input_current, 
                                                         simulation_time=400 * b2.ms)
plt.figure()
plot_tools.plot_voltage_and_current_traces(state_monitor, input_current)
print("nr of spikes: {}".format(spike_monitor.count[0]))
plt.figure()
AdEx.plot_adex_state(state_monitor)


# Adaption
input_current = LIF.input_factory.get_step_current(10, 250, 1. * b2.ms, 65.0 * b2.pA)
state_monitor, spike_monitor = AdEx.simulate_AdEx_neuron(tau_m=20 * b2.ms,
                                                         tau_w=100 * b2.ms,
                                                         a=0 * b2.nS,
                                                         b=5 * b2.pA,
                                                         v_reset=-55 * b2.mV,
                                                         I_stim=input_current, 
                                                         simulation_time=400 * b2.ms)
plt.figure()
plot_tools.plot_voltage_and_current_traces(state_monitor, input_current)
print("nr of spikes: {}".format(spike_monitor.count[0]))
plt.figure()
AdEx.plot_adex_state(state_monitor)


# Init. burst
input_current = LIF.input_factory.get_step_current(10, 250, 1. * b2.ms, 65.0 * b2.pA)
state_monitor, spike_monitor = AdEx.simulate_AdEx_neuron(tau_m=5 * b2.ms,
                                                         tau_w=100 * b2.ms,
                                                         a=0.5 * b2.nS,
                                                         b=7 * b2.pA,
                                                         v_reset=-51 * b2.mV,
                                                         I_stim=input_current, 
                                                         simulation_time=400 * b2.ms)
plt.figure()
plot_tools.plot_voltage_and_current_traces(state_monitor, input_current)
print("nr of spikes: {}".format(spike_monitor.count[0]))
plt.figure()
AdEx.plot_adex_state(state_monitor)


# Bursting
input_current = LIF.input_factory.get_step_current(10, 250, 1. * b2.ms, 65.0 * b2.pA)
state_monitor, spike_monitor = AdEx.simulate_AdEx_neuron(tau_m=5 * b2.ms,
                                                         tau_w=100 * b2.ms,
                                                         a=-0.5 * b2.nS,
                                                         b=7 * b2.pA,
                                                         v_reset=-46 * b2.mV,
                                                         I_stim=input_current, 
                                                         simulation_time=400 * b2.ms)
plt.figure()
plot_tools.plot_voltage_and_current_traces(state_monitor, input_current)
print("nr of spikes: {}".format(spike_monitor.count[0]))
plt.figure()
AdEx.plot_adex_state(state_monitor)






"""LIF networks"""
###############################################################################
###############################################################################
from neurodynex3.brunel_model import LIF_spiking_network
from neurodynex3.tools import plot_tools, spike_tools
import brian2 as b2

# Parameters of a single LIF neuron:
v_rest = 0. * b2.mV
v_reset = +10. * b2.mV
firing_threshold = +20. * b2.mV
membrane_time_scale = 20. * b2.ms   # tau_m
abs_refractory_period = 2.0 * b2.ms
sim_time = 250. * b2.ms

# Parameters of the network
'''Poisson rate of the external population is the only input'''
w0 = 0.1 * b2.mV  # w_ee=w_ie = w0 and = w_ei=w_ii = -g*w0
g = 4.  # balanced
connection_probability = 0.1
synaptic_delay = 1.5 * b2.ms
poisson_input_rate = 12. * b2.Hz   # Poisson rate of the external population
N_extern = 1000   # N_POISSON_INPUT

rate_monitor, spike_monitor, voltage_monitor, monitored_spike_idx = LIF_spiking_network.simulate_brunel_network(sim_time=sim_time)
plot_tools.plot_network_activity(rate_monitor, spike_monitor, voltage_monitor, spike_train_idx_list=monitored_spike_idx, t_min=0.*b2.ms)
spike_stats = spike_tools.get_spike_train_stats(spike_monitor, window_t_min= 100 *b2.ms)
plot_tools.plot_ISI_distribution(spike_stats, hist_nr_bins=100, xlim_max_ISI=sim_time)






"""Spatial Working Memory"""
###############################################################################
###############################################################################
from neurodynex3.working_memory_network import wm_model
from neurodynex3.tools import plot_tools
import brian2 as b2

wm_model.getting_started()

rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, w_profile = wm_model.simulate_wm(sim_time=800. * b2.ms, poisson_firing_rate=1.3 * b2.Hz, sigma_weight_profile=20., Jpos_excit2excit=1.6)
plot_tools.plot_network_activity(rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, t_min=0. * b2.ms)






"""Oja hebbian learning"""
###############################################################################
###############################################################################
import neurodynex3.ojas_rule.oja as oja
import matplotlib.pyplot as plt

cloud = oja.make_cloud(n=1000, ratio=.3, angle=60)     # Returns an oriented elliptic gaussian cloud of 2D points
wcourse = oja.learn(cloud, initial_angle=-20, eta=0.04)

plt.figure(figsize=(7,7))       # learning trace
plt.scatter(cloud[:, 0], cloud[:, 1], marker=".", alpha=.5)
plt.scatter(wcourse[:, 0], wcourse[:, 1], marker=".")
plt.plot(wcourse[-1, 0], wcourse[-1, 1], "or", markersize=7)
plt.axis('equal')
plt.title("The final weight vector is: ({},{})".format(round(wcourse[-1,0],3),
                                                        round(wcourse[-1,1]),3))






"""STDP learning"""
###############################################################################
###############################################################################
from time import time
import numpy as np
from random import sample
import brian2 as b2
from brian2.monitors import StateMonitor, SpikeMonitor, PopulationRateMonitor
from neurodynex3.tools import plot_tools, spike_tools

def poisson_generator(n_neuron, rate, t):
    return np.random.poisson(rate, (n_neuron,t))

N_input = 1000
N_recurrent = 4096
v_rest = -70 * b2.mV
v_reset = -65 * b2.mV
firing_threshold = -50 * b2.mV
membrane_time_scale = 8. * b2.ms
abs_refractory_period = 2.0 * b2.ms
synaptic_delay = 1.5 * b2.ms
Poisson_rate = np.random.rand(N_input)*1000
W = np.random.randn(N_input,N_recurrent)
M = np.random.randn(N_recurrent,N_recurrent)

start_time = time()
b2.start_scope()

# G
eqs = """dv/dt = -(v-v_rest) / membrane_time_scale : volt (unless refractory)"""
G = b2.NeuronGroup(N_recurrent, model=eqs, reset="v=v_reset", threshold="v>firing_threshold",
                   refractory=abs_refractory_period, method="linear")
G.v = v_rest  # set initial value
S_recurrnet = b2.Synapses(G, G, model='w : volt', on_pre='v_post += w', delay=synaptic_delay)
S_recurrnet.connect()
for i in range(N_recurrent):
    for j in range(N_recurrent):
        S_recurrnet.w[i,j] = M[i,j]*b2.mV
        
# P
P = b2.PoissonGroup(N_input, rates=Poisson_rate*b2.Hz)
S_feedforward = b2.Synapses(P, G, model='w : volt', on_pre='v_post += w', delay=synaptic_delay)
S_feedforward.connect()
for i in range(N_input):
    for j in range(N_recurrent):
        S_feedforward.w[i,j] = W[i,j]*b2.mV

# monitorss and run
idx_monitored_neurons = sample(range(N_recurrent), 3)
rate_monitor = PopulationRateMonitor(G)
spike_monitor = SpikeMonitor(G, record=True)
voltage_monitor = StateMonitor(G, "v", record=True)
b2.run(100*b2.ms)
end_time = time()
print('Cost time is:', end_time-start_time)


# plot
plot_tools.plot_network_activity(rate_monitor, spike_monitor, voltage_monitor, spike_train_idx_list=idx_monitored_neurons, t_min=0.*b2.ms)
spike_stats = spike_tools.get_spike_train_stats(spike_monitor, window_t_min= 100 *b2.ms)





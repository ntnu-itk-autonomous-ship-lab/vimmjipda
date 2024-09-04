"""
    Test module for the VIMMJIPDA interface class.
"""

import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.scenario_generator as sg
import colav_simulator.simulator as sim
import vimmjipda.vimmjipda_tracker_interface as vti

if __name__ == "__main__":
    tracker = vti.VIMMJIPDA()

    sbmpc_obj = ci.SBMPCWrapper()
    scenario_generator = sg.ScenarioGenerator()
    scenario_data_list = scenario_generator.generate_configured_scenarios()
    simulator = sim.Simulator()
    simulator.toggle_liveplot_visibility(True)
    output = simulator.run(scenario_data_list, colav_systems=[(0, sbmpc_obj)], trackers=[(0, tracker)])
    print("done")

#!/usr/bin/python3
from curses import wrapper

import argparse
import output_util
import race_car as rc
import race_track as rt
import race_simulator as rs
import rl_model_based as rl_mb
import rl_model_free as rl_mf
import statistics as st
import time


def parse_args():
    """
    Parse the arguments from command line
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--track", help="Specify the input file for the track")
    parser.add_argument("--reset_to_start", help="Return to the start of the "
                        "track upon crash. Default to resume from the closest "
                        "point of crash", action="store_true")
    parser.add_argument("--method", help="The method for learning. It's one of"
                        "value_iteration, q_learning or sarsa")
    return parser.parse_args()


def verify_args(args):
    """
    Verify dataset args are all legal args
    """
    
    if not args.track:
        raise Exception("Error: Track input file is required")


def test_simulation(simulator, printer):
    """
    Execute a classification task using the given configuration
    """

    max_rounds = 30
    cur_round = 0 
    while cur_round < max_rounds:
        finished = simulator.step()
        cur_states = simulator.current_states()
        printer.print("Round " + str(cur_round) + "\n" + cur_states)
        if finished:
            printer.print("Finished!", True)
            break
        cur_round += 1



def run_model_based_learning(simulator):
    """
    Run model based reinforcement learning using value iteration
    """
    rlmb = rl_mb.ReinformacementLearning(simulator)
    rlmb.value_iteration(0.1)
    return rlmb


def run_model_free_learning(simulator, use_q_learning = False, episodes = None):
    """
    Run model free reinforcement learning using either sarsa or q learning
    """
    rlmf = rl_mf.ReinformacementLearning(simulator, episodes)
    if use_q_learning:
        rlmf.q_learn()
    else: 
        rlmf.sarsa_learn()
    return rlmf


def run_simulator(rl_model, simulator, printer):
    """
    Run the simulation with the given model and simulator. 
    """
    max_time = 2000
    max_rounds = 10
    cur_round = 0
    finished_rounds = 0
    time_each_round = []
    time_each_success_round = []
    while cur_round < max_rounds:
        time = 1
        simulator.reset()
        print_cur_state(simulator, 0, 0, 0, printer, time)
        # Run the simulator until the car reached the finish line or exceeded
        # the number of steps allowed.
        while not simulator.has_finished() and time < max_time:
            ax, ay, quality = rl_model.get_action()
            simulator.run(ax, ay, randomness = True) 
            print_cur_state(simulator, ax, ay, quality, printer, time)
            time += 1

        # Record info about this round
        time_each_round.append(time - 1)
        if simulator.has_finished():
            finished_rounds += 1
            time_each_success_round.append(time - 1)
        # Increment cur_round for next iteration
        cur_round += 1

    return {
        "rounds_executed": max_rounds,
        "rounds_finished": finished_rounds,
        "time_each_round": time_each_round,
        "time_each_success_round": time_each_success_round,
    }


def run_experiment(args, simulator, printer, episodes = None):
    """
    Run experiment using the given param
    """
    start_time = time.time()
    rl_model = None
    if args.method == "value_iteration":
        rl_model = run_model_based_learning(simulator)
    elif args.method == "q_learning":
        rl_model = run_model_free_learning(simulator, True, episodes)
    else:
        rl_model = run_model_free_learning(simulator, False, episodes)
    
    training_time = time.time() - start_time
    results = [run_simulator(rl_model, simulator, printer)]
    return {
        "method": args.method,
        "rl_model": rl_model,
        "training_time": round(training_time, 4),
        "results": results
    }

def print_results(simulator, exp_results):
    """
    """
    output_util.print_csv_row(["------\n"])
    output_util.print_csv_row(["Track", simulator.track.get_name()])
    crash_policy = "Start" if simulator.is_reset_to_start() else "Nearest"
    output_util.print_csv_row(["Crash Rest", crash_policy])
    output_util.print_csv_row(["RL Method", exp_results.get("method")])

    rl_model = exp_results.get("rl_model")
    output_util.print_csv_row(["Model Attributes"])
    attributes = rl_model.get_attributes()
    for key in attributes:
        output_util.print_csv_row([key, attributes.get(key)])

    output_util.print_csv_row(["Training Time", exp_results.get("training_time")])
    
    all_results = exp_results.get("results")
    for results in all_results:
        output_util.print_csv_row(["Rounds Executed", results.get("rounds_executed")])
        output_util.print_csv_row(["Rounds Finished", results.get("rounds_finished")])

        time_each_round = results.get("time_each_round")
        avg_step = st.mean(time_each_round) if len(time_each_round) > 0 else 0
        output_util.print_csv_row(["Avg Steps", avg_step])
        
        time_each_success_round = results.get("time_each_success_round")
        avg_success_step = st.mean(time_each_success_round) if len(time_each_success_round) > 0 else 0
        output_util.print_csv_row(["Avg Success Steps", avg_success_step])
        output_util.print_csv_row(["------\n"])


def print_cur_state(simulator, ax, ay, quality, printer, time):
    """
    """
    if printer is None:
        return

    time = "Time " + str(time)
    [track_state, car_state] = simulator.current_states()
    action_quality = " ".join(
        ["Action:", str(ax), str(ay), "Quality:", str(round(quality, 3))])
    printer.print("\n".join(
        [time, car_state, action_quality, track_state]))


def main(stdscr = None):
    """ 
    The main function of the program
    """

    expisode_options = [3000, 2000, 1000] #, 200, 500, 1000, 1500]
    for episodes in expisode_options:
        # Start
        experiment_results = None
        simulator = None
        printer = None
        if stdscr:
            printer = output_util.ScreenPrinter(stdscr, delay = 0) 
        try:
            args = parse_args()
            verify_args(args)

            # Initialize the track from the input file
            track = rt.Track(args.track)
            car = rc.Car()
            simulator = rs.Simulator(track, car)
            simulator.set_reset_to_start(args.reset_to_start)
            # test_simulation(simulator, printer)
            experiment_results = run_experiment(
                args, simulator, printer, episodes)

        finally:
            if printer and printer.is_open:
                printer.close(3)

        print_results(simulator, experiment_results)
    # End


if __name__ == "__main__":
    # Configure whether to show progress
    should_print_progress = False
    if should_print_progress:
        wrapper(main)
    else:
        main()
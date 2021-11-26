import collections
import itertools as it
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pymysql

# %%
NORM_TIME = 1e6
LOADS = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def get_connection():
    return pymysql.connect(
        database="cerberus",
        host="localhost",
        port=3310,
        user="root",
        password="passwort_simulator"
    )


SystemConfig = collections.namedtuple("SystemConfig", ["topo_name", "algo_id"])
DemandConfig = collections.namedtuple("DemandConfig", ["flowgenname", "load", "bidi", "atgen", "fvgen_id", "congen_id"])

TOPO_TO_MARKERS = {
    "expander-16-40G": 'o',
    "rotornet-16-40G": 'x',
    "cerberus-16-0-4-12-40G": 's',
}

TOPO_TO_LINESTYLE = {
    "expander-16-40G": '-',
    "rotornet-16-40G": '--',
    "cerberus-16-0-4-12-40G": '-.'
}

TOPO_TO_COLOR = {
    "expander-16-40G": (1, 0, 0),
    "rotornet-16-40G": (0, 1, 0),
    "cerberus-16-0-4-12-40G": (0, 0, 1),
}

TOPO_TO_LABEL = {
    "expander-16-40G": 'expander-net',
    "rotornet-16-40G": 'rotor-net',
    "cerberus-16-0-4-12-40G": 'C-4R-12DA',
}

ALGO_TO_LABEL = {
    'SFF-TbT': 'TbT'
}


TOPOLOGIES_TO_DB_ID = {
    # FIXME adapt to your settings
    "expander-16-40G": (21,),
    "rotornet-16-40G": (19,),
    "cerberus-16-0-4-12-40G": (23,)
}
ALGOS_TO_DB_ID = {
    'Expander': 7,
    'RotorNet': 8,
    'Cerberus': 32
}

TOPO_CONFIGS = [
    SystemConfig(topo_name="rotornet-16-40G", algo_id="RotorNet"),
    SystemConfig(topo_name="expander-16-40G", algo_id='Expander'),
    SystemConfig(topo_name="cerberus-16-0-4-12-40G", algo_id='Cerberus'),
]
VOLUME_DATAMINING = 1

DEMAND_CONFIGS = [
    # Load translates to num. of generated flows. These simulations run 5s.
    # Therefore, load is i*5. Additionally, for 40G, multiply by 4.
    DemandConfig(
        flowgenname="RandomWithConnectionProbabilityFlowGeneratorConfiguration",
        load=i * 5 * 4, bidi=0, atgen="PoissonArrivalTimeGeneratorConfiguration",
        fvgen_id=VOLUME_DATAMINING, congen_id=None) for i in LOADS
]

# %% -------------------------------------------------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------------------------------------------------
FROM_DATABASE = False


# %%
def get_sim_ids(sysconfig, demconfig, sim_behavior=None, only_finished=True):
    query = "select simulation_id, fk_algorithm_id, fk_topology_id, t.name, " \
            " @uplinks := CAST(CONCAT_WS('', JSON_EXTRACT(t.parameter, '$.nr_links')," \
            " JSON_EXTRACT(t.parameter, '$.nr_rotorswitches'), " \
            " JSON_EXTRACT(t.parameter, '$.nr_static')+JSON_EXTRACT(t.parameter, '$.nr_rotor')+"\
            " JSON_EXTRACT(t.parameter, '$.nr_da')) AS INTEGER) as uplinks, " \
            " JSON_EXTRACT(t.parameter, '$.host_link_capacity') as host_cap, " \
            " REGEXP_SUBSTR(JSON_EXTRACT(t.parameter, '$.rotor_switch_configs'), 'reschedule_period=(\\d*)') as slot_size, " \
            " duration, f.name, " \
            " @calc_load := COALESCE(ROUND(f.number_flows/5088/@uplinks,2), 0) as calc_load " \
            " from simulation as s " \
            " inner join flowgenerator f on s.fk_flow_generator_id = f.flowgen_id " \
            f" {'inner join arrivaltimegenerator at on f.fk_arrival_time_generator_id = at.atgen_id ' if demconfig.atgen is not None else ''} " \
            " inner join topology t on s.fk_topology_id = t.topology_id " \
            "where " \
            f" fk_algorithm_id={ALGOS_TO_DB_ID[sysconfig.algo_id]} and " \
            f" fk_topology_id in {str(TOPOLOGIES_TO_DB_ID[sysconfig.topo_name]).replace(',)', ')')} and  " \
            f" f.name='{demconfig.flowgenname}'" \
            f" and simulation_builder_name in {sim_behavior} "
    if only_finished:
        query += f" and finished=1 "
    if demconfig.bidi:
        query += f" and f.bidirectional = {demconfig.bidi}"
    if demconfig.atgen:
        query += f" and at.name ='{demconfig.atgen}' "
    if demconfig.fvgen_id:
        query += f" and f.fk_flow_volume_generator_id = {demconfig.fvgen_id} "
    if demconfig.congen_id:
        query += f" and f.fk_connection_pair_generator_id = {demconfig.congen_id} "
    tmp = pd.read_sql(
        query,
        con=get_connection()
    ).set_index("calc_load")
    try:
        val = tmp.loc[demconfig.load]["simulation_id"]
        if type(val) == int:
            return [val]
        else:
            return val.values.tolist()
    except KeyError:
        return list()


def get_fct_data(sim_ids):
    query = f"select * from flowcompletiontime as fct " \
            f"inner join flow f on fct.fk_flow_id = f.flow_id " \
            f"inner join simulation s on fct.fk_simulation_id = s.simulation_id " \
            f"inner join flowgenerator on s.fk_flow_generator_id = flowgen_id " \
            f"where fk_simulation_id in {sim_ids} "
    return pd.read_sql(
        query, con=get_connection()
    ).set_index(
        ["fk_simulation_id", "fk_flow_id"]
    ).sort_index(inplace=False)


# %%
FCTS = dict()

if FROM_DATABASE:
    for syscfg, load in it.product(TOPO_CONFIGS, DEMAND_CONFIGS):
        simids = get_sim_ids(syscfg, load,
                             sim_behavior=("SimpleSimulationBuilderWithEventInsertion",),
                             only_finished=False)
        print(syscfg, load, simids)
        if len(simids) == 0:
            continue
        FCTS[(syscfg, load)] = get_fct_data(f"({','.join([str(s) for s in simids])})")
else:
    # TODO update
    PATH_TO_DATA = "/home/johannes/DATAS/cerberus/cerberus_flow_simulator_data/"

    # Read simulation configs:
    simconfigs = pd.read_csv(f"{PATH_TO_DATA}/simulations.csv")
    # Divide by base number of flows and num of uplinks
    simconfigs["calc_load"] = np.round(simconfigs["number_flows"] / 5088 / 16)
    simconfigs = simconfigs.reset_index().set_index(
        ["topology_id", "algorithm_id", "flowgen_name", "atgen_name", "fvgen_id", "calc_load"]
    ).sort_index(inplace=False)

    for syscfg, load in it.product(TOPO_CONFIGS, DEMAND_CONFIGS):
        simid = simconfigs.loc[
            TOPOLOGIES_TO_DB_ID[syscfg.topo_name][0],
            ALGOS_TO_DB_ID[syscfg.algo_id],
            load.flowgenname,
            load.atgen,
            load.fvgen_id,
            load.load
        ]["simulation_id"]

        FCTS[(syscfg, load)] = pd.read_csv(f"{PATH_TO_DATA}/fcts_{simid}.csv").set_index(
            ["fk_simulation_id", "flow_global_id"]
        ).sort_index(inplace=False)


# %% Plot served data for load 70%
def plot_line_acc_volume_finished_flows_time_system(fct_data, systems, demcfg, norm=1, sampling_factor=1000):
    agg_data = {}
    for syscfg in systems:
        try:
            thisdata = fct_data[(syscfg, demcfg)]
            simid = None
            max_samples = 0
            for i in thisdata.index.levels[0]:
                this_samples = len(thisdata.loc[i])
                if this_samples > max_samples:
                    max_samples = this_samples
                    simid = i

            if simid is None:
                print("Simid None")
                continue
        except KeyError as e:
            print(e)
            continue
        except IndexError as e:
            print(e)
            continue
        this_flows = thisdata
        this_flows["finish_time"] = 0
        this_flows["finish_time"] = (this_flows["arrival_time"] + this_flows["completion_time"])
        this_flows = this_flows.loc[simid].sort_values("finish_time")
        this_flows["finished_volume"] = (this_flows["volume"] * 2).cumsum()
        samples = len(this_flows)
        print(syscfg, simid, samples)
        indices = range(0, samples, int(samples / sampling_factor))

        plt.plot(
            this_flows.iloc[indices]["finish_time"],
            this_flows.iloc[indices]["finished_volume"] / norm,
            linestyle=TOPO_TO_LINESTYLE[syscfg.topo_name],
            color=TOPO_TO_COLOR[syscfg.topo_name],
            label=syscfg.algo_id,
        )

        agg_data[syscfg.algo_id] = {
            'time_us': this_flows.iloc[indices]["finish_time"].values.tolist(),
            'finished_volume_bits': this_flows.iloc[indices]["finished_volume"].values.tolist()
        }

    plt.xlabel("Simulation time [us]")
    plt.ylabel("Completed volume [bit]")
    return agg_data


# %%
demcfg = DEMAND_CONFIGS[5]
plt.figure(figsize=(1.5 * 3.48, 2.54))
agg_data = plot_line_acc_volume_finished_flows_time_system(
    FCTS,
    TOPO_CONFIGS,
    demcfg,
    norm=64 * 16 * 40e9
)

# Plot rate
plt.plot(
    [x * 1e6 for x in range(6)],
    [x * demcfg.load / 5 / 4 for x in range(6)],
    label="Offered load",
    color="k",
    linestyle=':'
)
plt.ylim(0, 0.5)
plt.xlim(0, 0.7e6)

plt.title(f"Load {demcfg.load / 4 / 5}")
ax = plt.gca()
ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
plt.legend(frameon=False, loc="lower left",
           bbox_to_anchor=(0, 0.4))
plt.tight_layout()
plt.savefig(f"line_acc_finished_volume_bidi_k16_poisson_40G.pdf")
plt.show()


# %%
def calculate_tp_load_systems(fct_data, systems, demands, time_lower_bound=0.3e6, time_upper_bound=1.05e6, norm=False,
                              norm_factor=5):
    agg_data = {}
    for syscfg in systems:
        tp_means = list()
        load_means = list()

        for demcfg in demands:
            try:
                thisdata = fct_data[(syscfg, demcfg)]
                simid = None
                max_samples = 0
                for i in thisdata.index.levels[0]:
                    this_samples = len(thisdata.loc[i])
                    if this_samples > max_samples:
                        max_samples = this_samples
                        simid = i
                if simid is None:
                    continue
            except KeyError as e:
                print(e)
                continue
            except IndexError as e:
                print(e)
                continue
            this_flows = thisdata
            this_flows["finish_time"] = 0
            this_flows["finish_time"] = (this_flows["arrival_time"] + this_flows["completion_time"])
            this_flows = this_flows.loc[simid].sort_values("finish_time")
            this_flows["finished_volume"] = (this_flows["volume"] * 2).cumsum()
            if time_upper_bound > 0:
                if this_flows["finish_time"].max() < time_upper_bound:
                    print(f"Data is not reaching end of time interval: {this_flows['finish_time'].max()}")
                    time_upper_bound = this_flows["finish_time"].max()

                idx_lower = (this_flows["finish_time"] - time_lower_bound).abs().idxmin()
                idx_upper = (this_flows["finish_time"] - time_upper_bound).abs().idxmin()
            else:
                idx_lower = this_flows["finish_time"].idxmin()
                idx_upper = this_flows["finish_time"].idxmax()

            val_lower = this_flows.loc[idx_lower]["finished_volume"]
            val_upper = this_flows.loc[idx_upper]["finished_volume"]
            time_lower = this_flows.loc[idx_lower]["finish_time"]
            time_upper = this_flows.loc[idx_upper]["finish_time"]

            # Norm with ideal for 16 Rotors, 64 ToRs and 40G links
            ideal = 1e3 if not norm else 16 * 40e3 * 64
            tp_means.append(
                (val_upper - val_lower) / (time_upper - time_lower) / ideal
            )
            print(
                f"t0: {time_lower}, v0: {val_lower}, t1: {time_upper}, v1: {val_upper}, slope: {tp_means[-1] * ideal}")
            load_means.append(demcfg.load / 4 / norm_factor)

        agg_data[syscfg.algo_id] = {
            'load': load_means,
            'finished_volume_bits': tp_means
        }
    return agg_data


# %%
agg_data = calculate_tp_load_systems(
    FCTS,
    TOPO_CONFIGS,
    DEMAND_CONFIGS,
    norm=True,
    norm_factor=5,  # Duration of simulation in seconds
    time_upper_bound=0.8e6,
    time_lower_bound=0.25e6
)

with open(f"throughput_poisson_long_40G.json", "w") as fd:
    json.dump(agg_data, fp=fd)


# %%
def plot_line_tp_load_systems(tp_data, systems):
    for syscfg in systems:
        try:
            this_data = tp_data[syscfg.algo_id]
        except KeyError:
            continue
        load_means = this_data["load"]
        tp_means = this_data["finished_volume_bits"]
        plt.plot(
            load_means,
            tp_means,
            linestyle=TOPO_TO_LINESTYLE[syscfg.topo_name],
            color=TOPO_TO_COLOR[syscfg.topo_name],
            label=syscfg.algo_id,
            marker=TOPO_TO_MARKERS[syscfg.topo_name]
        )


# %%
plt.figure()
plot_line_tp_load_systems(agg_data, TOPO_CONFIGS)

plt.ylim(0, 1.1)
plt.axhline(0.57, color=(1.0, 0, 0))
plt.axhline(0.64, color=(0, 1.0, 0))
plt.axhline(0.79, color=(0, 0, 1.0))

plt.xticks([0.2, 0.4, 0.6, 0.8], [f"{int(i * 100)}%" for i in [0.2, 0.4, 0.6, 0.8]])
plt.xlabel("Load")
plt.ylabel("Total throughput")
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels,
          frameon=False, loc="lower left",
          bbox_to_anchor=(0, 0.8)
          )
plt.savefig(f"line_throughput_poisson_long_40G.pdf")
plt.show()

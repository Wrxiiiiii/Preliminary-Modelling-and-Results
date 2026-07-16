import os
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "results_feeder_wide"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DER_NODES = [6, 12]

# Scenario grid for the full 10 x 10 study
SLACK_VOLTAGE_MIN = 0.98
SLACK_VOLTAGE_MAX = 1.03
N_SLACK_STEPS = 10

LOAD_FACTOR_MIN = 0.2
LOAD_FACTOR_MAX = 0.8
N_LOAD_STEPS = 10

# Thermal constraints remain disabled at this stage.
USE_THERMAL_CONSTRAINT = True

# Reactive-power control settings used in the comparison case.
QMAX_FRAC = 0.25
LOCAL_VOLTAGE_THRESHOLD = 1.04
VOLTAGE_BINDING_TOL = 1e-3
THERMAL_BINDING_TOL = 1e-3
CAPACITY_BINDING_TOL = 1e-4

print("Output directory:", OUTPUT_DIR)
print("Number of scenarios:", N_SLACK_STEPS * N_LOAD_STEPS)

N = 13
slack = 0

base_l_P = np.array([
    0, 0.2, 0, 0.4, 0.17, 0.23, 1.155,
    0, 0.17, 0.843, 0, 0.17, 0.128
])

base_l_Q = np.array([
    0, 0.116, 0, 0.29, 0.125, 0.132,
    0.66, 0, 0.151, 0.462, 0, 0.08, 0.086
])

v_min = 0.95
v_max = 1.05

r = np.array([
[0, 0.007547918, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0.0041, 0, 0.007239685, 0, 0.007547918, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0.0041, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0.004343811, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.003773959, 0.003773959, 0, 0.004322245, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00434686, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004343157, 0.01169764],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

x = np.array([
[0, 0.022173236, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0.0064, 0, 0.007336076, 0, 0.022173236, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0.0064, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0.004401645, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.011086618, 0.011086618, 0, 0.004433667, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002430473, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004402952, 0.004490848],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

I_max = np.array([
[0, 3.0441, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1.4178, 0, 0.9591, 0, 3.0441, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 3.1275, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0.9591, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 3.0441, 3.1275, 0, 0.9591, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.37193, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9591, 1.2927],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

A = np.array([
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

edges = [(i, j) for i in range(N) for j in range(N) if A[i, j] == 1]

children = {i: [] for i in range(N)}
parent = {}
for i, j in edges:
    children[i].append(j)
    parent[j] = i

print("Number of feeder branches:", len(edges))

pv_capacity = np.zeros(N)
pv_capacity[12] = 1.5
pv_capacity[6] = 3.5

def get_pv_available():
    """Full PV availability is assumed in this network-stress scenario grid."""
    pv_available = np.zeros(N)
    for j in DER_NODES:
        pv_available[j] = pv_capacity[j]
    return pv_available

def create_scenario_grid():
    slack_values = np.linspace(
        SLACK_VOLTAGE_MIN,
        SLACK_VOLTAGE_MAX,
        N_SLACK_STEPS,
    )
    load_values = np.linspace(
        LOAD_FACTOR_MIN,
        LOAD_FACTOR_MAX,
        N_LOAD_STEPS,
    )

    rows = []
    scenario = 0
    for slack_voltage_pu in slack_values:
        for load_factor in load_values:
            rows.append({
                "scenario": scenario,
                "slack_voltage_pu": float(slack_voltage_pu),
                "load_factor": float(load_factor),
            })
            scenario += 1

    return pd.DataFrame(rows)

# ============================================================
# Feeder-wide OE solver
# ============================================================

def solve_feeder_wide_export_oe(
    l_P,
    l_Q,
    pv_available,
    slack_voltage_pu,
    use_reactive_support=False,
    qmax_frac=0.0,
    verbose=False,
):
    """
    Maximise total accommodated DER production.

    The function returns branch flows as well as DER production and
    voltages so that voltage, thermal, and capacity constraints can
    be diagnosed after the optimisation.
    """
    P = cp.Variable((N, N))
    Q = cp.Variable((N, N))
    v = cp.Variable(N)

    p_der = cp.Variable(N, nonneg=True)
    q_der = cp.Variable(N)

    constraints = [
        v[slack] == slack_voltage_pu ** 2,
        v >= v_min ** 2,
        v <= v_max ** 2,
    ]

    for j in range(N):
        if j in DER_NODES:
            constraints += [p_der[j] <= pv_available[j]]
        else:
            constraints += [
                p_der[j] == 0,
                q_der[j] == 0,
            ]

    # Local voltage-based Q droop.
    if use_reactive_support:
        for j in DER_NODES:
            qmax_j = qmax_frac * pv_capacity[j]
            constraints += [
                q_der[j] == qmax_j * (
                    (v_max ** 2 + v_min ** 2 - 2.0 * v[j])
                    / (v_max ** 2 - v_min ** 2)
                )
            ]
    else:
        constraints += [q_der == 0]

    edge_set = set(edges)

    for i in range(N):
        for j in range(N):
            if (i, j) not in edge_set:
                constraints += [
                    P[i, j] == 0,
                    Q[i, j] == 0,
                ]

    for j in range(1, N):
        i = parent[j]

        downstream_P = cp.sum([P[j, k] for k in children[j]])
        downstream_Q = cp.sum([Q[j, k] for k in children[j]])

        net_P = l_P[j] - p_der[j]
        net_Q = l_Q[j] - q_der[j]

        constraints += [
            P[i, j] == net_P + downstream_P,
            Q[i, j] == net_Q + downstream_Q,
            v[j] == v[i] - 2.0 * (
                r[i, j] * P[i, j]
                + x[i, j] * Q[i, j]
            ),
        ]

        if USE_THERMAL_CONSTRAINT:
            constraints += [
                cp.norm(cp.hstack([P[i, j], Q[i, j]]), 2)
                <= I_max[i, j]
            ]

    oe_total = cp.sum(p_der[DER_NODES])
    problem = cp.Problem(cp.Maximize(oe_total), constraints)

    try:
        problem.solve(solver=cp.CLARABEL, verbose=verbose)
    except Exception:
        problem.solve(solver=cp.SCS, verbose=verbose)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return {
            "status": problem.status,
            "OE_total_MW": np.nan,
            "p_der": np.full(N, np.nan),
            "q_der": np.full(N, np.nan),
            "voltage": np.full(N, np.nan),
            "P": np.full((N, N), np.nan),
            "Q": np.full((N, N), np.nan),
        }

    voltage = np.sqrt(np.maximum(v.value, 0.0))

    return {
        "status": problem.status,
        "OE_total_MW": float(np.sum(p_der.value[DER_NODES])),
        "p_der": np.asarray(p_der.value, dtype=float),
        "q_der": np.asarray(q_der.value, dtype=float),
        "voltage": voltage,
        "P": np.asarray(P.value, dtype=float),
        "Q": np.asarray(Q.value, dtype=float),
    }


# ============================================================
# Constraint diagnosis
# ============================================================

def diagnose_binding_constraints(result, pv_available):
    """
    Determine which constraints are active at the optimum.

    This is used to explain why Node 12 reduces production after its
    local voltage rises above approximately 1.04 p.u.
    """
    if result["status"] not in ["optimal", "optimal_inaccurate"]:
        return {
            "upper_voltage_binding": False,
            "upper_voltage_binding_nodes": "",
            "lower_voltage_binding": False,
            "lower_voltage_binding_nodes": "",
            "thermal_binding": False,
            "thermal_binding_branches": "",
            "node_6_capacity_binding": False,
            "node_12_capacity_binding": False,
            "likely_limiting_mechanism": "Infeasible",
        }

    voltage = result["voltage"]
    P = result["P"]
    Q = result["Q"]
    p_der = result["p_der"]

    upper_nodes = [
        i for i in range(N)
        if np.isfinite(voltage[i])
        and abs(voltage[i] - v_max) <= VOLTAGE_BINDING_TOL
    ]

    lower_nodes = [
        i for i in range(N)
        if np.isfinite(voltage[i])
        and abs(voltage[i] - v_min) <= VOLTAGE_BINDING_TOL
    ]

    thermal_branches = []
    for i, j in edges:
        apparent_power = float(np.hypot(P[i, j], Q[i, j]))
        if (
            I_max[i, j] > 0
            and abs(apparent_power - I_max[i, j])
            <= THERMAL_BINDING_TOL
        ):
            thermal_branches.append(f"{i}-{j}")

    node_6_capacity_binding = bool(
        abs(p_der[6] - pv_available[6]) <= CAPACITY_BINDING_TOL
    )
    node_12_capacity_binding = bool(
        abs(p_der[12] - pv_available[12]) <= CAPACITY_BINDING_TOL
    )

    # A readable diagnosis focused on Node 12.
    if 12 in upper_nodes:
        likely_cause = "Node 12 upper-voltage constraint"
    elif upper_nodes:
        likely_cause = (
            "Upper-voltage constraint at node(s) "
            + ",".join(str(i) for i in upper_nodes)
        )
    elif thermal_branches:
        likely_cause = (
            "Thermal constraint on branch(es) "
            + ",".join(thermal_branches)
        )
    elif node_12_capacity_binding:
        likely_cause = "Node 12 PV availability"
    elif node_6_capacity_binding:
        likely_cause = "Node 6 PV availability"
    else:
        likely_cause = "Combined network constraints / non-unique allocation"

    return {
        "upper_voltage_binding": bool(upper_nodes),
        "upper_voltage_binding_nodes": ",".join(
            str(i) for i in upper_nodes
        ),
        "lower_voltage_binding": bool(lower_nodes),
        "lower_voltage_binding_nodes": ",".join(
            str(i) for i in lower_nodes
        ),
        "thermal_binding": bool(thermal_branches),
        "thermal_binding_branches": ",".join(thermal_branches),
        "node_6_capacity_binding": node_6_capacity_binding,
        "node_12_capacity_binding": node_12_capacity_binding,
        "likely_limiting_mechanism": likely_cause,
    }


# ============================================================
# Run one complete case
# ============================================================

def run_case(
    case_name,
    use_reactive_support,
    qmax_frac,
):
    """
    Run all 100 scenarios for one Q-control setting.
    """
    case_output_dir = os.path.join(OUTPUT_DIR, case_name)
    os.makedirs(case_output_dir, exist_ok=True)

    scenario_df = create_scenario_grid()
    pv_available = get_pv_available()

    scenario_rows = []
    der_rows = []

    for _, row in scenario_df.iterrows():
        scenario = int(row["scenario"])
        slack_voltage_pu = float(row["slack_voltage_pu"])
        load_factor = float(row["load_factor"])

        l_P_s = base_l_P * load_factor
        l_Q_s = base_l_Q * load_factor

        result = solve_feeder_wide_export_oe(
            l_P=l_P_s,
            l_Q=l_Q_s,
            pv_available=pv_available,
            slack_voltage_pu=slack_voltage_pu,
            use_reactive_support=use_reactive_support,
            qmax_frac=qmax_frac,
        )

        diagnosis = diagnose_binding_constraints(
            result,
            pv_available,
        )

        voltage = result["voltage"]
        p_der = result["p_der"]
        q_der = result["q_der"]

        scenario_rows.append({
            "case": case_name,
            "reactive_support": use_reactive_support,
            "qmax_frac": qmax_frac,
            "scenario": scenario,
            "slack_voltage_pu": slack_voltage_pu,
            "load_factor": load_factor,
            "status": result["status"],
            "OE_total_MW": result["OE_total_MW"],
            "max_voltage_pu": (
                np.nan
                if np.all(np.isnan(voltage))
                else float(np.nanmax(voltage))
            ),
            "min_voltage_pu": (
                np.nan
                if np.all(np.isnan(voltage))
                else float(np.nanmin(voltage))
            ),
            **diagnosis,
        })

        for j in DER_NODES:
            production = p_der[j]
            available = pv_available[j]
            local_load = l_P_s[j]
            local_voltage = voltage[j]

            der_rows.append({
                "case": case_name,
                "reactive_support": use_reactive_support,
                "qmax_frac": qmax_frac,
                "scenario": scenario,
                "slack_voltage_pu": slack_voltage_pu,
                "load_factor": load_factor,
                "status": result["status"],
                "DER_node": j,
                "PV_available_MW": available,
                "P_DER_MW": production,
                "alpha": (
                    np.nan
                    if available <= 0 or np.isnan(production)
                    else production / available
                ),
                "local_load_MW": local_load,
                "net_nodal_export_MW": (
                    np.nan
                    if np.isnan(production)
                    else production - local_load
                ),
                "curtailment_MW": (
                    np.nan
                    if np.isnan(production)
                    else available - production
                ),
                "local_voltage_pu": local_voltage,
                "voltage_above_1p04_pu": (
                    np.nan
                    if np.isnan(local_voltage)
                    else max(
                        local_voltage - LOCAL_VOLTAGE_THRESHOLD,
                        0.0,
                    )
                ),
                "Q_DER_MVAr": q_der[j],
                "upper_voltage_binding": diagnosis[
                    "upper_voltage_binding"
                ],
                "upper_voltage_binding_nodes": diagnosis[
                    "upper_voltage_binding_nodes"
                ],
                "thermal_binding": diagnosis["thermal_binding"],
                "thermal_binding_branches": diagnosis[
                    "thermal_binding_branches"
                ],
                "likely_limiting_mechanism": diagnosis[
                    "likely_limiting_mechanism"
                ],
            })

    scenario_results_df = pd.DataFrame(scenario_rows)
    der_results_df = pd.DataFrame(der_rows)

    feasible_scenarios = scenario_results_df[
        scenario_results_df["status"].isin(
            ["optimal", "optimal_inaccurate"]
        )
    ].copy()

    feasible_der = der_results_df[
        der_results_df["status"].isin(
            ["optimal", "optimal_inaccurate"]
        )
    ].copy()

    scenario_file = os.path.join(
        case_output_dir,
        f"{case_name}_scenario_results.csv",
    )
    der_file = os.path.join(
        case_output_dir,
        f"{case_name}_DER_results.csv",
    )

    scenario_results_df.to_csv(scenario_file, index=False)
    der_results_df.to_csv(der_file, index=False)

    node12 = feasible_der[
        feasible_der["DER_node"] == 12
    ].copy()

    node12_above_104 = node12[
        node12["local_voltage_pu"] > LOCAL_VOLTAGE_THRESHOLD
    ].copy()

    summary_df = pd.DataFrame([{
        "case": case_name,
        "reactive_support": use_reactive_support,
        "qmax_frac": qmax_frac,
        "number_of_scenarios": len(scenario_results_df),
        "optimal_cases": len(feasible_scenarios),
        "infeasible_cases": (
            len(scenario_results_df) - len(feasible_scenarios)
        ),
        "mean_total_OE_MW": feasible_scenarios[
            "OE_total_MW"
        ].mean(),
        "mean_node_6_production_MW": feasible_der.loc[
            feasible_der["DER_node"] == 6,
            "P_DER_MW",
        ].mean(),
        "mean_node_12_production_MW": node12[
            "P_DER_MW"
        ].mean(),
        "mean_node_12_alpha": node12["alpha"].mean(),
        "node_12_scenarios_above_1p04": len(node12_above_104),
        "node_12_mean_production_above_1p04_MW": (
            node12_above_104["P_DER_MW"].mean()
        ),
        "upper_voltage_binding_cases": int(
            feasible_scenarios[
                "upper_voltage_binding"
            ].sum()
        ),
        "thermal_binding_cases": int(
            feasible_scenarios["thermal_binding"].sum()
        ),
    }])

    summary_file = os.path.join(
        case_output_dir,
        f"{case_name}_summary.csv",
    )
    summary_df.to_csv(summary_file, index=False)

    # --------------------------------------------------------
    # Diagnostic plots
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))
    for j in DER_NODES:
        node_data = feasible_der[
            feasible_der["DER_node"] == j
        ]
        plt.scatter(
            node_data["local_voltage_pu"],
            node_data["P_DER_MW"],
            alpha=0.75,
            label=f"Node {j}",
        )

    plt.axvline(
        LOCAL_VOLTAGE_THRESHOLD,
        linestyle="--",
        label="1.04 p.u. threshold",
    )
    plt.axvline(
        v_max,
        linestyle=":",
        label="1.05 p.u. upper limit",
    )
    plt.xlabel("Local voltage, p.u.")
    plt.ylabel("DER production, MW")
    plt.title(f"DER Production vs Local Voltage: {case_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            case_output_dir,
            f"{case_name}_production_vs_voltage.png",
        ),
        dpi=300,
    )
    plt.close()

    # Node 12 production, with marker colour showing local demand.
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        node12["local_voltage_pu"],
        node12["P_DER_MW"],
        c=node12["local_load_MW"],
        alpha=0.85,
    )
    plt.colorbar(
        scatter,
        label="Node 12 local demand, MW",
    )
    plt.axvline(
        LOCAL_VOLTAGE_THRESHOLD,
        linestyle="--",
        label="1.04 p.u. threshold",
    )
    plt.axvline(
        v_max,
        linestyle=":",
        label="1.05 p.u. upper limit",
    )
    plt.xlabel("Node 12 local voltage, p.u.")
    plt.ylabel("Node 12 production, MW")
    plt.title(
        f"Node 12 Production vs Voltage and Demand: {case_name}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            case_output_dir,
            f"{case_name}_node12_voltage_demand.png",
        ),
        dpi=300,
    )
    plt.close()

    # Constraint class shown by separate marker groups.
    plt.figure(figsize=(9, 6))
    for mechanism, group in node12.groupby(
        "likely_limiting_mechanism"
    ):
        plt.scatter(
            group["local_voltage_pu"],
            group["P_DER_MW"],
            alpha=0.80,
            label=mechanism,
        )

    plt.axvline(
        LOCAL_VOLTAGE_THRESHOLD,
        linestyle="--",
        label="1.04 p.u. threshold",
    )
    plt.axvline(
        v_max,
        linestyle=":",
        label="1.05 p.u. upper limit",
    )
    plt.xlabel("Node 12 local voltage, p.u.")
    plt.ylabel("Node 12 production, MW")
    plt.title(
        f"Node 12 Binding-Constraint Diagnosis: {case_name}"
    )
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            case_output_dir,
            f"{case_name}_node12_constraint_diagnosis.png",
        ),
        dpi=300,
    )
    plt.close()

    print("\n" + "=" * 70)
    print("Case:", case_name)
    print("=" * 70)
    print(summary_df.to_string(index=False))

    if not node12_above_104.empty:
        print("\nNode 12 scenarios with local voltage above 1.04 p.u.")
        print(
            node12_above_104[
                [
                    "scenario",
                    "slack_voltage_pu",
                    "load_factor",
                    "P_DER_MW",
                    "local_voltage_pu",
                    "local_load_MW",
                    "upper_voltage_binding_nodes",
                    "thermal_binding_branches",
                    "likely_limiting_mechanism",
                ]
            ].to_string(index=False)
        )

    return {
        "scenario_results": scenario_results_df,
        "der_results": der_results_df,
        "summary": summary_df,
        "output_dir": case_output_dir,
    }


# ============================================================
# Local linear control for Node 12
# ============================================================

def calculate_regression_metrics(y_true, y_pred):
    residual = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))

    total_variation = float(
        np.sum((y_true - np.mean(y_true)) ** 2)
    )
    unexplained_variation = float(np.sum(residual ** 2))

    r_squared = (
        np.nan
        if total_variation <= 1e-12
        else 1.0
        - unexplained_variation / total_variation
    )

    return {
        "RMSE_MW": rmse,
        "MAE_MW": mae,
        "R_squared": r_squared,
    }


def fit_node12_local_control(
    case_results,
    train_fraction=0.8,
    random_seed=42,
):
    """
    Fit and evaluate two local control functions for Node 12.

    Inputs available to the local controller:
        - local voltage at Node 12;
        - local active-power demand at Node 12.

    Model 1:
        P12 = b0 + b1*V12 + b2*Demand12

    Model 2:
        P12 = b0 + b1*V12 + b2*Demand12
              + b3*max(V12 - 1.04, 0)

    The models are fitted on a training set and evaluated on a
    separate test set. This avoids reporting only an in-sample fit.
    """
    der_results_df = case_results["der_results"]
    case_name = str(
        der_results_df["case"].dropna().iloc[0]
    )
    output_dir = case_results["output_dir"]

    data = der_results_df[
        (der_results_df["DER_node"] == 12)
        & der_results_df["status"].isin(
            ["optimal", "optimal_inaccurate"]
        )
    ].dropna(
        subset=[
            "local_voltage_pu",
            "local_load_MW",
            "P_DER_MW",
        ]
    ).copy().reset_index(drop=True)

    if len(data) < 10:
        raise ValueError(
            "At least 10 feasible Node 12 observations are "
            "required for the train/test analysis."
        )

    # --------------------------------------------------------
    # Reproducible 80/20 train-test split
    # --------------------------------------------------------
    rng = np.random.default_rng(random_seed)
    shuffled_indices = rng.permutation(len(data))

    number_train = int(np.floor(train_fraction * len(data)))
    number_train = min(max(number_train, 1), len(data) - 1)

    train_indices = shuffled_indices[:number_train]
    test_indices = shuffled_indices[number_train:]

    data["dataset_split"] = "test"
    data.loc[train_indices, "dataset_split"] = "train"

    voltage = data["local_voltage_pu"].to_numpy(dtype=float)
    demand = data["local_load_MW"].to_numpy(dtype=float)
    production = data["P_DER_MW"].to_numpy(dtype=float)

    voltage_excess = np.maximum(
        voltage - LOCAL_VOLTAGE_THRESHOLD,
        0.0,
    )

    X_linear = np.column_stack([
        np.ones(len(data)),
        voltage,
        demand,
    ])

    X_threshold = np.column_stack([
        np.ones(len(data)),
        voltage,
        demand,
        voltage_excess,
    ])

    # --------------------------------------------------------
    # Fit both models using training observations only
    # --------------------------------------------------------
    beta_linear, *_ = np.linalg.lstsq(
        X_linear[train_indices],
        production[train_indices],
        rcond=None,
    )

    beta_threshold, *_ = np.linalg.lstsq(
        X_threshold[train_indices],
        production[train_indices],
        rcond=None,
    )

    prediction_linear = np.clip(
        X_linear @ beta_linear,
        0.0,
        pv_capacity[12],
    )

    prediction_threshold = np.clip(
        X_threshold @ beta_threshold,
        0.0,
        pv_capacity[12],
    )

    data["P12_predicted_linear_MW"] = prediction_linear
    data[
        "P12_predicted_threshold_MW"
    ] = prediction_threshold

    # --------------------------------------------------------
    # Training and testing metrics
    # --------------------------------------------------------
    train_metrics_linear = calculate_regression_metrics(
        production[train_indices],
        prediction_linear[train_indices],
    )
    test_metrics_linear = calculate_regression_metrics(
        production[test_indices],
        prediction_linear[test_indices],
    )

    train_metrics_threshold = calculate_regression_metrics(
        production[train_indices],
        prediction_threshold[train_indices],
    )
    test_metrics_threshold = calculate_regression_metrics(
        production[test_indices],
        prediction_threshold[test_indices],
    )

    coefficients_df = pd.DataFrame([
        {
            "case": case_name,
            "model": "linear_voltage_and_demand",
            "train_fraction": train_fraction,
            "random_seed": random_seed,
            "number_training_observations": len(train_indices),
            "number_testing_observations": len(test_indices),
            "intercept": beta_linear[0],
            "beta_voltage": beta_linear[1],
            "beta_local_demand": beta_linear[2],
            "beta_voltage_above_1p04": 0.0,
            "train_RMSE_MW": train_metrics_linear["RMSE_MW"],
            "train_MAE_MW": train_metrics_linear["MAE_MW"],
            "train_R_squared": train_metrics_linear["R_squared"],
            "test_RMSE_MW": test_metrics_linear["RMSE_MW"],
            "test_MAE_MW": test_metrics_linear["MAE_MW"],
            "test_R_squared": test_metrics_linear["R_squared"],
        },
        {
            "case": case_name,
            "model": "threshold_linear_control",
            "train_fraction": train_fraction,
            "random_seed": random_seed,
            "number_training_observations": len(train_indices),
            "number_testing_observations": len(test_indices),
            "intercept": beta_threshold[0],
            "beta_voltage": beta_threshold[1],
            "beta_local_demand": beta_threshold[2],
            "beta_voltage_above_1p04": beta_threshold[3],
            "train_RMSE_MW": (
                train_metrics_threshold["RMSE_MW"]
            ),
            "train_MAE_MW": (
                train_metrics_threshold["MAE_MW"]
            ),
            "train_R_squared": (
                train_metrics_threshold["R_squared"]
            ),
            "test_RMSE_MW": (
                test_metrics_threshold["RMSE_MW"]
            ),
            "test_MAE_MW": (
                test_metrics_threshold["MAE_MW"]
            ),
            "test_R_squared": (
                test_metrics_threshold["R_squared"]
            ),
        },
    ])

    coefficients_df.to_csv(
        os.path.join(
            output_dir,
            (
                f"{case_name}_node12_local_control_"
                "train_test_coefficients.csv"
            ),
        ),
        index=False,
    )

    data.to_csv(
        os.path.join(
            output_dir,
            (
                f"{case_name}_node12_local_control_"
                "train_test_predictions.csv"
            ),
        ),
        index=False,
    )

    # --------------------------------------------------------
    # Figure 1: test-set actual versus predicted
    # --------------------------------------------------------
    y_test = production[test_indices]
    linear_test = prediction_linear[test_indices]
    threshold_test = prediction_threshold[test_indices]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        y_test,
        linear_test,
        alpha=0.80,
        label="Linear LR, test set",
    )
    plt.scatter(
        y_test,
        threshold_test,
        alpha=0.80,
        label="Threshold LR, test set",
    )

    plot_min = float(
        min(
            y_test.min(),
            linear_test.min(),
            threshold_test.min(),
        )
    )
    plot_max = float(
        max(
            y_test.max(),
            linear_test.max(),
            threshold_test.max(),
        )
    )

    plt.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        linestyle="--",
        label="Perfect prediction",
    )
    plt.xlabel("OPF Node 12 production, MW")
    plt.ylabel("Predicted Node 12 production, MW")
    plt.title(
        f"Node 12 Local Control Test Performance: {case_name}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            (
                f"{case_name}_node12_local_control_"
                "test_actual_vs_predicted.png"
            ),
        ),
        dpi=300,
    )
    plt.close()

    # --------------------------------------------------------
    # Figure 2: OPF vs LR vs threshold LR on the test set
    # --------------------------------------------------------
    test_plot = data.loc[test_indices].copy()
    test_plot = test_plot.sort_values("local_voltage_pu")

    plt.figure(figsize=(9, 6))
    plt.scatter(
        test_plot["local_voltage_pu"],
        test_plot["P_DER_MW"],
        s=55,
        alpha=0.85,
        label="OPF production, test set",
    )
    plt.plot(
        test_plot["local_voltage_pu"],
        test_plot["P12_predicted_linear_MW"],
        marker="o",
        label="Linear LR prediction",
    )
    plt.plot(
        test_plot["local_voltage_pu"],
        test_plot["P12_predicted_threshold_MW"],
        marker="s",
        label="Threshold LR prediction",
    )
    plt.axvline(
        LOCAL_VOLTAGE_THRESHOLD,
        linestyle="--",
        label="1.04 p.u. threshold",
    )
    plt.axvline(
        v_max,
        linestyle=":",
        label="1.05 p.u. upper limit",
    )
    plt.xlabel("Node 12 local voltage, p.u.")
    plt.ylabel("Node 12 production, MW")
    plt.title(
        f"OPF vs Local LR Controls on Test Scenarios: {case_name}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            (
                f"{case_name}_node12_OPF_vs_LR_"
                "threshold_LR_test.png"
            ),
        ),
        dpi=300,
    )
    plt.close()

    # --------------------------------------------------------
    # Figure 3: smooth control functions at mean local demand
    # --------------------------------------------------------
    voltage_grid = np.linspace(
        data["local_voltage_pu"].min(),
        data["local_voltage_pu"].max(),
        200,
    )
    representative_demand = float(
        data["local_load_MW"].mean()
    )
    voltage_grid_excess = np.maximum(
        voltage_grid - LOCAL_VOLTAGE_THRESHOLD,
        0.0,
    )

    X_grid_linear = np.column_stack([
        np.ones(len(voltage_grid)),
        voltage_grid,
        np.full(len(voltage_grid), representative_demand),
    ])

    X_grid_threshold = np.column_stack([
        np.ones(len(voltage_grid)),
        voltage_grid,
        np.full(len(voltage_grid), representative_demand),
        voltage_grid_excess,
    ])

    grid_prediction_linear = np.clip(
        X_grid_linear @ beta_linear,
        0.0,
        pv_capacity[12],
    )
    grid_prediction_threshold = np.clip(
        X_grid_threshold @ beta_threshold,
        0.0,
        pv_capacity[12],
    )

    plt.figure(figsize=(9, 6))
    plt.scatter(
        data["local_voltage_pu"],
        data["P_DER_MW"],
        c=data["local_load_MW"],
        alpha=0.55,
        label="OPF scenarios",
    )
    plt.plot(
        voltage_grid,
        grid_prediction_linear,
        linewidth=2,
        label=(
            "Linear LR at mean demand "
            f"({representative_demand:.3f} MW)"
        ),
    )
    plt.plot(
        voltage_grid,
        grid_prediction_threshold,
        linewidth=2,
        label=(
            "Threshold LR at mean demand "
            f"({representative_demand:.3f} MW)"
        ),
    )
    plt.colorbar(label="Node 12 local demand, MW")
    plt.axvline(
        LOCAL_VOLTAGE_THRESHOLD,
        linestyle="--",
        label="1.04 p.u. threshold",
    )
    plt.axvline(
        v_max,
        linestyle=":",
        label="1.05 p.u. upper limit",
    )
    plt.xlabel("Node 12 local voltage, p.u.")
    plt.ylabel("Node 12 production, MW")
    plt.title(
        f"Local Control Functions for Node 12: {case_name}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            (
                f"{case_name}_node12_local_control_"
                "smooth_curves.png"
            ),
        ),
        dpi=300,
    )
    plt.close()

    print(
        "\nNode 12 local-control train/test regression:",
        case_name,
    )
    print(coefficients_df.to_string(index=False))

    return coefficients_df, data


# ============================================================
# Compare no-Q and Q-control cases
# ============================================================

def compare_no_q_and_q(
    no_q_results,
    q_results,
):
    combined_summary = pd.concat(
        [
            no_q_results["summary"],
            q_results["summary"],
        ],
        ignore_index=True,
    )
    combined_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "no_Q_vs_Q_summary.csv",
        ),
        index=False,
    )

    combined_der = pd.concat(
        [
            no_q_results["der_results"],
            q_results["der_results"],
        ],
        ignore_index=True,
    )

    node12 = combined_der[
        (combined_der["DER_node"] == 12)
        & combined_der["status"].isin(
            ["optimal", "optimal_inaccurate"]
        )
    ].copy()

    plt.figure(figsize=(9, 6))
    for case_name, case_data in node12.groupby("case"):
        plt.scatter(
            case_data["local_voltage_pu"],
            case_data["P_DER_MW"],
            alpha=0.75,
            label=case_name,
        )

    plt.axvline(
        LOCAL_VOLTAGE_THRESHOLD,
        linestyle="--",
        label="1.04 p.u. threshold",
    )
    plt.axvline(
        v_max,
        linestyle=":",
        label="1.05 p.u. upper limit",
    )
    plt.xlabel("Node 12 local voltage, p.u.")
    plt.ylabel("Node 12 production, MW")
    plt.title("Node 12 Production: Without and With Q Control")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "node12_no_Q_vs_Q_control.png",
        ),
        dpi=300,
    )
    plt.close()

    # Scenario-by-scenario production comparison.
    node12_pivot = node12.pivot(
        index="scenario",
        columns="case",
        values="P_DER_MW",
    )

    plt.figure(figsize=(10, 6))
    for case_name in node12_pivot.columns:
        plt.plot(
            node12_pivot.index,
            node12_pivot[case_name],
            label=case_name,
        )

    plt.xlabel("Scenario")
    plt.ylabel("Node 12 production, MW")
    plt.title("Scenario-by-Scenario Node 12 Production")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "node12_production_by_scenario_no_Q_vs_Q.png",
        ),
        dpi=300,
    )
    plt.close()

    print("\nNo-Q and Q-control comparison")
    print(combined_summary.to_string(index=False))


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    print("\nRunning feeder-wide OE without Q control...")
    no_q_results = run_case(
        case_name="no_Q_control",
        use_reactive_support=False,
        qmax_frac=0.0,
    )

    print("\nRunning feeder-wide OE with Q control...")
    q_results = run_case(
        case_name="with_Q_control_Qmax25pct",
        use_reactive_support=True,
        qmax_frac=QMAX_FRAC,
    )

    fit_node12_local_control(no_q_results)
    fit_node12_local_control(q_results)

    compare_no_q_and_q(
        no_q_results,
        q_results,
    )

    print("\nFinished all analyses.")

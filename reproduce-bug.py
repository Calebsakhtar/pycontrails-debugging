import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pycontrails

from pathlib import Path
from pycontrails import Flight
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.models.humidity_scaling import ConstantHumidityScaling
from pycontrails.physics import units

def run_from_flight(met_filepath):
    attrs = {
        "flight_id": "COMPARISON",
        "aircraft_type": "B738",
        "wingspan": 34.32
    }
    
    df = pd.read_csv("inputs/pycontrails-flight.csv")
    fl = Flight(data=df, attrs=attrs)

    # Load met and override met data
    file = open("inputs/met/met.pkl", 'rb')
    met = pickle.load(file)
    file.close()
    met.data = xr.open_dataset(met_filepath ,engine='netcdf4')

    # Load rad (not in use)
    file = open("inputs/met/rad.pkl", 'rb')
    rad = pickle.load(file)
    file.close()

    params = {
    "process_emissions": False,
    "verbose_outputs": True,
    "humidity_scaling": ConstantHumidityScaling(rhi_adj=1),
    }
    
    cocip = Cocip(met=met, rad=rad, params=params)
    fl_out = cocip.eval(source=fl) # One flight

    df = pd.DataFrame.from_dict(fl_out.data)

    return df, cocip.contrail

def process_and_save_outputs(contrail, filepath, idx = 14):
    # Create an empty list for the rows
    row_list =[]

    # Create empty lists for the results
    times = []
    N = []
    n =[]
    alt_m = []
    rhi = []
    iwc = []
    T = []
    rho_air_dry = []
    p = []
    SH = []
    shear = []
    area_eff = []
    tau = []
    width = []
    depth = []
    segment_length = []
    plume_mass = []
    q_sat = []

    if contrail is not None:
        # Iterate over each row 
        for index, row in contrail.iterrows():
            if row.waypoint == idx:
                row_list.append(row)

                times.append(row.age.total_seconds() / 60. / 60.)
                N.append(row.n_ice_per_m)
                n.append(row.n_ice_per_vol)
                alt_m.append(row.altitude)
                rhi.append(row.rhi)
                iwc.append(row.iwc)
                T.append(row.air_temperature)
                rho_air_dry.append(row.rho_air)
                p.append(row.air_pressure)
                SH.append(row.specific_humidity)
                shear.append(row.dsn_dz)
                area_eff.append(row.area_eff)
                tau.append(row.tau_contrail)
                width.append(row.width)
                depth.append(row.depth)
                segment_length.append(row.segment_length)
                plume_mass.append(row.plume_mass_per_m)
                q_sat.append(row.q_sat)

    # Convert the lists to numpy arrays
    times = np.array(times)
    N = np.array(N)
    n = np.array(n)
    alt_m = np.array(alt_m)
    rhi = np.array(rhi)
    iwc = np.array(iwc)
    T = np.array(T)
    rho_air_dry = np.array(rho_air_dry)
    p = np.array(p)
    SH = np.array(SH)
    shear = np.array(shear)
    area_eff = np.array(area_eff)
    tau = np.array(tau)
    width = np.array(width)
    depth = np.array(depth)
    segment_length = np.array(segment_length)
    plume_mass = np.array(plume_mass)
    q_sat = np.array(q_sat)

    rho_air_moist = rho_air_dry * ( SH / (1 - SH) + 1)
    I = iwc * plume_mass
    intOD = tau * width
    water_mass = plume_mass * (iwc + q_sat)

    data = {
        "Time Since Formation, h": times,
        "N, # / m": N,
        "n, # / m^3": n,
        "Altitude, m": alt_m,
        "RHi, %": rhi * 100,
        "IWC, kg of ice / kg of moist air": iwc,
        "Air Temperature, K": T,
        "Dry Air Density, kg / m^3": rho_air_dry,
        "Moist Air Density, kg / m^3": rho_air_moist,
        "Air Pressure, Pa": p,
        "Specific Humidity, Ratio": SH,
        "Shear, 1 / s": shear,
        "Effective Area, m^2": area_eff,
        "Optical Depth, ---": tau,
        "I, kg of ice / m": I,
        "Width, m": width,
        "Depth, m": depth,
        "Segment Length, m": segment_length,
        "Integrated Optical Depth, m": intOD,
        "Plume Water Mass, kg / m": water_mass,
    }

    DF = pd.DataFrame.from_dict(data)
    DF.to_csv(filepath)

    return data

def test_RHi():
    oppath_model = "outputs/met"
    Path(oppath_model).mkdir(parents=True, exist_ok=True)

    ippath = "inputs/met"
    for filename in os.listdir(ippath):
        if filename.endswith('.nc') and not filename.startswith('tristan'):
            RHi = filename[:3]
            print("\n**************")
            print(f"Running case with RHi {RHi}%")
            df, contrail = run_from_flight(met_filepath=f"{ippath}/{filename}")
            process_and_save_outputs(
                contrail = contrail,
                idx = 14,
                filepath = f"{oppath_model}/{RHi}throughout-OP.csv"
            )
            print("**************\n")

    Ns = []
    ns = []
    Is = []
    ts = []
    intODs = []
    As = []
    ODs = []
    widths = []
    depths = []
    RHis = []
    alts = []
    rhis = []
    shs = []
    Ts = []
    rhos = []
    H2Os = []
    for filename in os.listdir(oppath_model):
        if filename.endswith('.csv'):
            df = pd.read_csv(f"{oppath_model}/{filename}")
            ts.append(df["Time Since Formation, h"].values)
            Ns.append(df["N, # / m"].values)
            Is.append(df["I, kg of ice / m"].values)
            intODs.append(df["Integrated Optical Depth, m"].values)
            ODs.append(df["Optical Depth, ---"].values)
            As.append(df["Effective Area, m^2"].values)
            widths.append(df["Width, m"].values)
            depths.append(df["Depth, m"].values)
            alts.append(df["Altitude, m"].values)
            ns.append(df["n, # / m^3"].values)
            rhis.append(df["RHi, %"].values)
            shs.append(df['Specific Humidity, Ratio'].values)
            Ts.append(df['Air Temperature, K'].values)
            rhos.append(df['Dry Air Density, kg / m^3'].values)
            H2Os.append(df["Plume Water Mass, kg / m"].values)
            RHis.append(filename[:3])

    oppath = "outputs/"
    Path(oppath).mkdir(parents=True, exist_ok=True)

    for i in range(len(RHis)):
        plt.plot(ts[i],Ns[i],label = "RHi = " + str(RHis[i]) + " %")
    plt.ylabel("Total Ice Particles, #/m")
    plt.xlabel("t, h")
    plt.yscale('log')
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_N_vs_t.png")
    plt.close()

    for i in range(len(RHis)):
        plt.plot(ts[i],Is[i],label = "RHi = " + str(RHis[i]) + " %")   
    plt.ylabel("Total Ice Mass, kg/m")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_I_vs_t.png")
    plt.close()

    for i in range(len(RHis)):
        plt.plot(ts[i],intODs[i],label = "RHi = " + str(RHis[i]) + " %")   
    plt.ylabel("Optical Depth * Width, m")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_intOD_vs_t.png")
    plt.close()

    for i in range(len(RHis)):
        plt.plot(ts[i],ODs[i],label = "RHi = " + str(RHis[i]) + " %")   
    plt.ylabel("Optical Depth")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.yscale('log')
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_OD_vs_t.png")
    plt.close()

    for i in range(len(RHis)):
        plt.plot(ts[i],As[i],label = "RHi = " + str(RHis[i]) + " %")   
    plt.ylabel("Area, m^2")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_area_vs_t.png")
    plt.close()

    for i in range(len(RHis)):
        plt.plot(ts[i],widths[i],label = "RHi = " + str(RHis[i]) + " %")   
    plt.ylabel("Width, m")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_width_vs_t.png")
    plt.close()

    for i in range(len(RHis)):
        plt.plot(ts[i],depths[i],label = "RHi = " + str(RHis[i]) + " %")   
    plt.ylabel("Depth, m")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_depth_vs_t.png")
    plt.close()

    for i in range(len(RHis)):
        plt.plot(ts[i],alts[i],label = "RHi = " + str(RHis[i]) + " %")   
    plt.ylabel("Altitude, m")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_alt_vs_t.png")
    plt.close()

    for i in range(len(RHis)):
        plt.plot(ts[i],ns[i],label = "RHi = " + str(RHis[i]) + " %")   
    plt.ylabel("n, #/m^3")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.yscale('log')
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_n_per_m3_vs_t.png")
    plt.close()

    for i in range(len(RHis)):
        plt.plot(ts[i],rhis[i],label = "RHi = " + str(RHis[i]) + " %")   
    plt.ylabel("RHi, %")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_RHi_vs_t.png")
    plt.close()

    for i in range(len(RHis)):
        plt.plot(ts[i],shs[i],label = "RHi = " + str(RHis[i]) + " %")   
    plt.ylabel("Specific Humidity")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_SH_vs_t.png")
    plt.close()

    for i in range(len(RHis)):
        plt.plot(ts[i],Ts[i],label = "RHi = " + str(RHis[i]) + " %")   
    plt.ylabel("Air Temperature, K")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_T_vs_t.png")
    plt.close()

    for i in range(len(RHis)):
        plt.plot(ts[i],rhos[i],label = "RHi = " + str(RHis[i]) + " %")   
    plt.ylabel("Air Density, kg / m^3")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_rho_vs_t.png")
    plt.close()

    for i in range(len(RHis)):
        plt.plot(ts[i],H2Os[i],label = "RHi = " + str(RHis[i]) + " %")   
    plt.ylabel("Plume Water Mass, kg / m")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_H20_vs_t.png")
    plt.close()

def test_tristan():
    oppath_model = "outputs/met"
    Path(oppath_model).mkdir(parents=True, exist_ok=True)

    ippath = "inputs/met"
    for filename in os.listdir(ippath):
        if filename == "tristan-met.nc":
            df, contrail = run_from_flight(met_filepath=f"{ippath}/{filename}")
            process_and_save_outputs(
                contrail = contrail,
                idx = 14,
                filepath = f"{oppath_model}/tristan.csv"
            )
            print("**************\n")

    Ns = []
    ns = []
    Is = []
    ts = []
    intODs = []
    As = []
    ODs = []
    widths = []
    depths = []
    RHis = []
    alts = []
    rhis = []
    shs = []
    Ts = []
    rhos = []
    H2Os = []

    for filename in os.listdir(oppath_model):
        if filename == "tristan.csv":
            df = pd.read_csv(f"{oppath_model}/{filename}")
            ts = df["Time Since Formation, h"].values
            Ns = df["N, # / m"].values
            Is = df["I, kg of ice / m"].values
            intODs = df["Integrated Optical Depth, m"].values
            ODs = df["Optical Depth, ---"].values
            As = df["Effective Area, m^2"].values
            widths = df["Width, m"].values
            depths = df["Depth, m"].values
            alts = df["Altitude, m"].values
            ns = df["n, # / m^3"].values
            rhis = df["RHi, %"].values
            shs = df['Specific Humidity, Ratio'].values
            Ts = df['Air Temperature, K'].values
            rhos = df['Dry Air Density, kg / m^3'].values
            H2Os = df["Plume Water Mass, kg / m"].values

    oppath = "outputs/tristan/"
    Path(oppath).mkdir(parents=True, exist_ok=True)


    plt.plot(ts,Ns)
    plt.ylabel("Total Ice Particles, #/m")
    plt.xlabel("t, h")
    plt.yscale('log')
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_tristan_N_vs_t.png")
    plt.close()

    
    plt.plot(ts,Is)   
    plt.ylabel("Total Ice Mass, kg/m")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_tristan_I_vs_t.png")
    plt.close()

    
    plt.plot(ts,intODs)   
    plt.ylabel("Optical Depth * Width, m")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_tristan_intOD_vs_t.png")
    plt.close()

    
    plt.plot(ts,ODs)   
    plt.ylabel("Optical Depth")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.yscale('log')
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_tristan_OD_vs_t.png")
    plt.close()

    
    plt.plot(ts,As)   
    plt.ylabel("Area, m^2")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_tristan_area_vs_t.png")
    plt.close()

    
    plt.plot(ts,widths)   
    plt.ylabel("Width, m")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_tristan_width_vs_t.png")
    plt.close()

    
    plt.plot(ts,depths)   
    plt.ylabel("Depth, m")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_tristan_depth_vs_t.png")
    plt.close()

    
    plt.plot(ts,alts)   
    plt.ylabel("Altitude, m")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_tristan_alt_vs_t.png")
    plt.close()

    
    plt.plot(ts,ns)   
    plt.ylabel("n, #/m^3")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.yscale('log')
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_tristan_n_per_m3_vs_t.png")
    plt.close()

    
    plt.plot(ts,rhis)   
    plt.ylabel("RHi, %")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_tristan_RHi_vs_t.png")
    plt.close()

    
    plt.plot(ts,shs)   
    plt.ylabel("Specific Humidity")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_tristan_SH_vs_t.png")
    plt.close()

    
    plt.plot(ts,Ts)   
    plt.ylabel("Air Temperature, K")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_tristan_T_vs_t.png")
    plt.close()

    
    plt.plot(ts,rhos)   
    plt.ylabel("Air Density, kg / m^3")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_tristan_rho_vs_t.png")
    plt.close()

    
    plt.plot(ts,H2Os)   
    plt.ylabel("Plume Water Mass, kg / m")
    plt.xlabel("t, h")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_tristan_H20_vs_t.png")
    plt.close()

    plt.plot(rhis,alts)   
    plt.ylabel("Altitude, m")
    plt.xlabel("Relative Humidity wrt Ice, %")
    plt.legend(reverse=True)
    plt.tight_layout(pad=2.0)
    plt.savefig(oppath + "fig_tristan_alt_vs_rhi.png")
    plt.close()


if __name__ == "__main__":
    print(pycontrails.__version__)
    test_RHi()
    test_tristan()

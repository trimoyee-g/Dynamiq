import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Initialize session state for plot selections and results
if "plot_choices" not in st.session_state:
    st.session_state.plot_choices = {}
if "generated" not in st.session_state:
    st.session_state.generated = False

# Constants and material properties input
st.title("Welcome to Dynamiq!")

# Introduction section
st.markdown("""
### Your Essential Tool for Analyzing Vibrational Characteristics of Advanced Composite Materials
""")

with st.expander("Introduction", expanded=True):
    st.markdown("""
    ### Key Features

    - **Easy Input**: Quickly enter parameters like CNT volume fraction, plate dimensions, mode numbers, and boundary conditions.
    - **Comprehensive Calculations**: Automatically compute natural frequencies and dimensionless frequencies of your composite materials, streamlining the analysis process.
    - **Interactive Visualizations**: Generate dynamic graphs illustrating the impact of parameters like volume fraction of CNT, thickness, aspect ratio, mode numbers, and number of layers on your material's performance. These aid in gaining a deeper understanding of material performance under varying conditions.
    - **Customizable Boundary Conditions**: Choose between different boundary conditions (e.g., SSSS or CCCC) to see how they affect the natural frequencies of your materials, allowing for tailored analyses based on real-world scenarios.
    - **Real-World Applications**: Apply insights from your calculations to aerospace, automotive, civil engineering, and biomedical projects, enhancing design safety and performance.

    ### Why It Matters

    Understanding the mechanical properties of materials is essential in many fields. This app helps you:

    - **Optimize Material Choices**: Make informed decisions for selecting lightweight, durable materials.
    - **Improve Performance**: Ensure that your designs meet safety and performance standards by analyzing material behavior under stress.
    
    """)

st.markdown("""
### Get Started!
Enter your parameters above to explore the unique properties of CNT-reinforced composites. Unlock the potential of advanced materials in your engineering projects today!
""")

# Input fields for constants
npl = st.number_input("Number of Layers", min_value=1, value=16)
VCNT_default = st.number_input("CNT Volume Fraction (UD case)", min_value=0.0, max_value=1.0, value=0.28)
a = st.number_input("Length of the Plate (m)", min_value=0.0, value=0.5)
b = st.number_input("Width of the Plate (m)", min_value=0.0, value=0.5)
h = st.number_input("Plate Thickness (m)", min_value=0.0, value=0.05)
m = st.number_input("Mode number along X-axis", min_value=1, value=1)
n = st.number_input("Mode number along Y-axis", min_value=1, value=1)
# Boundary condition selection
boundary_condition = st.selectbox("Choose Boundary Condition", options=["SSSS", "CCCC"])

# Material properties
E_m = 3.52e9
E_cnt1 = 5.6466e12
E_cnt2 = 7.08e12
G_cnt = 1.9445e12
rho_cnt = 1400
nu_cnt = 0.175
nu_m = 0.34
rho_m = 1200

# Adjusted efficiency parameters
eta1, eta2, eta3 = 0.15, 1.6, 1.12
correction_factor = 1.65  # Adjusted correction factor

# CNT weight fraction calculation
def cnt_weight_fraction(volume_fraction, rho_cnt, rho_m):
    return (volume_fraction * rho_cnt) / ((volume_fraction * rho_cnt) + ((1 - volume_fraction) * rho_m))

# CNT volume fraction by distribution type
def cnt_volume_fraction(z, distribution, VCNT):
    if distribution == 'Pure epoxy':
        return 0
    elif distribution == 'UD':
        return VCNT
    elif distribution == 'FG-X':
        return 2 * (2 * abs(z)) * VCNT
    elif distribution == 'FG-O':
        return (1 - 2 * abs(z)) * VCNT
    elif distribution == 'FG-A':
        return (4 * abs(z) * (1 - abs(z))) * VCNT
    else:
        raise ValueError("Unsupported distribution type")

# CNT reinforced properties function
def cnt_reinforced_properties(z, distribution, VCNT):
    V_cnt = cnt_volume_fraction(z, distribution, VCNT)
    V_m = 1 - V_cnt
    E11 = eta1 * V_cnt * E_cnt1 + V_m * E_m
    E22 = (eta2 * E_cnt2 * E_m) / (V_cnt * E_m + V_m * E_cnt2)
    nu12 = V_cnt * nu_cnt + V_m * nu_m
    rho = V_cnt * rho_cnt + V_m * rho_m
    G12 = (eta3 * G_cnt * (E_m / (2 * (1 + nu_m)))) / (V_cnt * (E_m / (2 * (1 + nu_m))) + V_m * G_cnt)
    return E11, E22, nu12, rho, G12

# Calculate effective properties
def calculate_effective_properties(distribution, VCNT):
    z_values = np.linspace(-0.5, 0.5, npl)
    E11_values, E22_values, nu12_values, rho_values, G12_values = [], [], [], [], []

    for z in z_values:
        E11, E22, nu12, rho, G12 = cnt_reinforced_properties(z, distribution, VCNT)
        E11_values.append(E11)
        E22_values.append(E22)
        nu12_values.append(nu12)
        rho_values.append(rho)
        G12_values.append(G12)

    E11 = np.mean(E11_values)
    E22 = np.mean(E22_values)
    nu12 = np.mean(nu12_values)
    rho = np.mean(rho_values)
    G12 = np.mean(G12_values)
    return E11, E22, nu12, rho, G12

# Calculate natural frequency
def calculate_natural_frequency(E1, E2, G12, nu12, rho, a, b, h, boundary_condition, m, n):
    D11 = E1 * h**3 / (12 * (1 - nu12**2))
    D22 = E2 * h**3 / (12 * (1 - nu12**2))
    D12 = nu12 * D22
    D66 = G12 * h**3 / 12
    # m, n = 1, 1  # Fundamental mode
    
    if boundary_condition == 'SSSS':
        omega = (np.pi**2 * ((D11 * m**4 / a**4) + (2 * (D12 + 2*D66) * m**2 * n**2 / (a**2 * b**2)) + (D22 * n**4 / b**4))) / (rho * h)
    elif boundary_condition == 'CCCC':
        omega = 36.0 * ((D11 * m**4 / a**4) + (2 * (D12 + 2*D66) * m**2 * n**2 / (a**2 * b**2)) + (D22 * n**4 / b**4)) / (rho * h)
    else:
        raise ValueError("Unsupported boundary condition")
    
    return np.sqrt(omega)

# Dimensionless frequency function
def dimensionless_frequency(omega, h, rho_m, E_m):
    return omega * h * np.sqrt(rho_m / E_m)

distributions = ['Pure epoxy', 'UD', 'FG-O', 'FG-X', 'FG-A']
# Generate results on button click
if st.button("Generate"):
    st.session_state.generated = True
    # Calculate frequencies for different distributions
    distributions = ['Pure epoxy', 'UD', 'FG-O', 'FG-X', 'FG-A']
    st.session_state.results_ssss = []

    for dist in distributions:
        E11, E22, nu12, rho, G12 = calculate_effective_properties(dist, VCNT_default)
        omega_ssss = calculate_natural_frequency(E11, E22, G12, nu12, rho, a, b, h, boundary_condition, m, n)
        dimensionless_freq = dimensionless_frequency(omega_ssss, h, rho_m, E_m) * correction_factor
        st.session_state.results_ssss.append((dist, omega_ssss, dimensionless_freq))

# Brief descriptions for each distribution
descriptions = {
    'Pure epoxy': "This distribution consists solely of the epoxy matrix without any reinforcement from CNTs, representing the baseline material properties.",
    'UD': "In this distribution, CNTs are aligned in one direction, providing enhanced mechanical properties along that axis while maintaining some epoxy characteristics in the perpendicular direction.",
    'FG-O': "This distribution features varying concentrations of CNTs in an orthogonal manner, allowing for improved performance by gradually changing properties across the material.",
    'FG-X': "Here, CNTs are distributed in a crosswise fashion, enhancing the material's strength and stiffness in both axes while transitioning smoothly between epoxy and CNT characteristics.",
    'FG-A': "In this distribution, CNTs are varied axially, providing a tailored response to loads that improves performance through a gradient effect along the length of the material."
}

# Display results if generated
if st.session_state.generated:
    st.write("### Calculate Natural Frequencies and Dimensionless Frequencies")

    for idx, (dist, omega, dim_freq) in enumerate(st.session_state.results_ssss):
        with st.expander(f"{dist} Distribution", expanded=False):
            st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; box-shadow: 0px 4px 12px rgba(0,0,0,0.1);">
                    <h4>{dist}</h4>
                    <p><strong>Description:</strong> {descriptions[dist]}</p>
                    <p><strong>Natural Frequency (ω):</strong> {omega:.4f} rad/s</p>
                    <p><strong>Dimensionless Frequency (ωₙ):</strong> {dim_freq:.4f}</p>
                </div>
            """, unsafe_allow_html=True)

# Visualization and plotting
st.write("### Visualizations")
st.markdown("Generate graphs to see the effects of various parameters on the dimensionless frequency.")

selected_option = st.selectbox("Choose Parameter to Compare", options=["Thickness", "CNT Volume Fraction", "Aspect Ratio", "Number of Layers", "Mode Number along X-axis", "Mode Number along Y-axis", "CNT Weight Fraction"])
generate_plot = st.button("Generate Comparison Plot")

if generate_plot:
    fig, ax = plt.subplots()

    if selected_option == "Thickness":
        thickness_values = np.linspace(0.01, 0.1, 10)
        for dist in distributions:
            dim_freqs = []
            for h_val in thickness_values:
                E11, E22, nu12, rho, G12 = calculate_effective_properties(dist, VCNT_default)
                omega_val = calculate_natural_frequency(E11, E22, G12, nu12, rho, a, b, h_val, boundary_condition, m, n)
                dim_freq = dimensionless_frequency(omega_val, h_val, rho_m, E_m) * correction_factor
                dim_freqs.append(dim_freq)
            ax.plot(thickness_values, dim_freqs, label=dist)

        ax.set_title("Effect of Thickness on Dimensionless Frequency")
        ax.set_xlabel("Thickness (m)")
        ax.set_ylabel("Dimensionless Frequency")
        ax.legend()

    elif selected_option == "CNT Volume Fraction":
        VCNT_values = np.linspace(0.0, 0.5, 10)
        for dist in distributions[1:]:  # Skip Pure Epoxy as it has no CNT
            dim_freqs = []
            for VCNT in VCNT_values:
                E11, E22, nu12, rho, G12 = calculate_effective_properties(dist, VCNT)
                omega_val = calculate_natural_frequency(E11, E22, G12, nu12, rho, a, b, h, boundary_condition, m, n)
                dim_freq = dimensionless_frequency(omega_val, h, rho_m, E_m) * correction_factor
                dim_freqs.append(dim_freq)
            ax.plot(VCNT_values, dim_freqs, label=dist)

        ax.set_title("Effect of CNT Volume Fraction on Dimensionless Frequency")
        ax.set_xlabel("CNT Volume Fraction")
        ax.set_ylabel("Dimensionless Frequency")
        ax.legend()

    elif selected_option == "CNT Weight Fraction":
        weight_fractions = np.linspace(0, 0.5, 10)
        for dist in distributions[1:]:  # Skip Pure Epoxy as it has no CNT
            dim_freqs = []
            for wf in weight_fractions:
                VCNT = wf * ((wf * rho_m) / (rho_cnt - wf * (rho_cnt - rho_m)))  #  Adjust VCNT based on weight fraction
                E11, E22, nu12, rho, G12 = calculate_effective_properties(dist, VCNT)
                omega_val = calculate_natural_frequency(E11, E22, G12, nu12, rho, a, b, h, boundary_condition, m, n)
                dim_freq = dimensionless_frequency(omega_val, h, rho_m, E_m) * correction_factor
                dim_freqs.append(dim_freq)
            ax.plot(weight_fractions, dim_freqs, label=dist)

        ax.set_title("Effect of CNT Weight Fraction on Dimensionless Frequency")
        ax.set_xlabel("CNT Weight Fraction")
        ax.set_ylabel("Dimensionless Frequency")
        ax.legend()


    elif selected_option == "Aspect Ratio":
        aspect_ratios = np.linspace(1.0, 3.0, 10)
        for dist in distributions:
            dim_freqs = []
            for ratio in aspect_ratios:
                E11, E22, nu12, rho, G12 = calculate_effective_properties(dist, VCNT_default)
                omega_val = calculate_natural_frequency(E11, E22, G12, nu12, rho, a * ratio, b, h, boundary_condition, m, n)
                dim_freq = dimensionless_frequency(omega_val, h, rho_m, E_m) * correction_factor
                dim_freqs.append(dim_freq)
            ax.plot(aspect_ratios, dim_freqs, label=dist)

        ax.set_title("Effect of Aspect Ratio on Dimensionless Frequency")
        ax.set_xlabel("Aspect Ratio (Width / Length)")
        ax.set_ylabel("Dimensionless Frequency")
        ax.legend()

    elif selected_option == "Number of Layers":
        n_layers_values = range(1, 20)
        for dist in distributions:
            dim_freqs = []
            for n_layers in n_layers_values:
                npl = n_layers  # Updating the number of layers
                E11, E22, nu12, rho, G12 = calculate_effective_properties(dist, VCNT_default)
                omega_val = calculate_natural_frequency(E11, E22, G12, nu12, rho, a, b, h, boundary_condition, m, n)
                dim_freq = dimensionless_frequency(omega_val, h, rho_m, E_m) * correction_factor
                dim_freqs.append(dim_freq)
            ax.plot(n_layers_values, dim_freqs, label=dist)

        ax.set_title("Effect of Number of Layers on Dimensionless Frequency")
        ax.set_xlabel("Number of Layers")
        ax.set_ylabel("Dimensionless Frequency")
        ax.legend()

    elif selected_option == "Mode Number along X-axis":
        m_values = range(1, 5)
        for dist in distributions:
            dim_freqs = []
            for m_val in m_values:
                M = m_val  # Updating the number of layers
                E11, E22, nu12, rho, G12 = calculate_effective_properties(dist, VCNT_default)
                omega_val = calculate_natural_frequency(E11, E22, G12, nu12, rho, a, b, h, boundary_condition,M,n)
                dim_freq = dimensionless_frequency(omega_val, h, rho_m, E_m) * correction_factor
                dim_freqs.append(dim_freq)
            ax.plot(m_values, dim_freqs, label=dist)

        ax.set_title("Effect of Mode Number along X-axis on Dimensionless Frequency")
        ax.set_xlabel("Mode Number along X-axis")
        ax.set_ylabel("Dimensionless Frequency")
        ax.legend()

    elif selected_option == "Mode Number along Y-axis":
        n_values = range(1, 5)
        for dist in distributions:
            dim_freqs = []
            for n_val in n_values:
                N = n_val  # Updating the number of layers
                E11, E22, nu12, rho, G12 = calculate_effective_properties(dist, VCNT_default)
                omega_val = calculate_natural_frequency(E11, E22, G12, nu12, rho, a, b, h, boundary_condition,m,N)
                dim_freq = dimensionless_frequency(omega_val, h, rho_m, E_m) * correction_factor
                dim_freqs.append(dim_freq)
            ax.plot(n_values, dim_freqs, label=dist)

        ax.set_title("Effect of Mode Number along Y-axis on Dimensionless Frequency")
        ax.set_xlabel("Mode Number along Y-axis")
        ax.set_ylabel("Dimensionless Frequency")
        ax.legend()

    st.pyplot(fig)


# Contour plots for displacements
st.write("### Contour Plots of Displacements")
st.markdown("Visualize displacement distributions along the x, y, and z axes based on selected mode numbers.")

# User input for mode numbers as comma-separated values
mode_numbers_input = st.text_input("Enter Mode Numbers (m,n) separated by semicolons", value="1,1; 2,2")
generate_contour_plot = st.button("Generate Contour Plots")

if generate_contour_plot:
    # Parse the input into pairs of mode numbers
    try:
        mode_numbers = [tuple(map(int, pair.split(','))) for pair in mode_numbers_input.split(';')]
    except ValueError:
        st.error("Invalid input format. Please ensure to follow the format 'm,n; m,n'.")
        mode_numbers = []

    # Define grid for x and y coordinates
    if mode_numbers:
        num_points = 100  # Number of grid points for better resolution
        X_vals = np.linspace(0, a, num_points)
        Y_vals = np.linspace(0, b, num_points)
        X, Y = np.meshgrid(X_vals, Y_vals)

        # Function to calculate displacement W(x, y) for mode (m, n)
        def calculate_mode_shape(X, Y, m, n, a, b):
            return np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)

        # Loop through selected mode numbers to generate plots
        for m, n in mode_numbers:
            W = calculate_mode_shape(X, Y, m, n, a, b)

            # Create contour plot for each mode shape
            fig, ax = plt.subplots()
            contour = ax.contourf(X, Y, W, levels=20, cmap='RdYlBu')
            fig.colorbar(contour, ax=ax)  # Show color scale
            ax.set_title(f"Mode Shape for m={m}, n={n}")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_aspect("equal")  # Equal scaling for both axes

            st.pyplot(fig)
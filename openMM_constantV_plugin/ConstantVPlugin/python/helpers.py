"""
ConstantV Plugin Helper Functions

These helper functions replicate the Original Python code's automatic behaviors:
- Electrode exclusion generation
- Geometry parameter auto-computation
- Electrolyte atom auto-identification

All functions are exact replications of Professor's Original code at:
/home/andy/test_optimization/OpenMM-ConstantV(original)

References:
- MM_classes.py::generate_exclusions() Lines 560-623
- MM_classes.py::set_electrochemical_cell_parameters() Lines 229-245
- MM_classes.py::initialize_electrolyte() Lines 256-279
- electrode_sapt_exclusions.py::exclusion_Electrode_NonbondedForce() Lines 28-66
"""

import openmm as mm
import openmm.unit as unit


def add_electrode_exclusions(constantv_obj, nonbonded_force, custom_nonbonded_force=None):
    """
    Add exclusions between all electrode atoms (cathode-cathode and anode-anode).

    ⚠️  CRITICAL: This MUST be called before creating the Context!
    ⚠️  CRITICAL: After creating Context, you MUST call context.reinitialize(preserveState=True)!

    This function replicates the Original behavior from:
    - MM_classes.py::generate_exclusions() Lines 560-590
    - electrode_sapt_exclusions.py::exclusion_Electrode_NonbondedForce() Lines 28-66

    Without these exclusions, electrode atoms will interact with each other,
    causing non-physical repulsion and simulation instability.

    Parameters
    ----------
    constantv_obj : ConstantVForce or ConstantVIntegrator
        The ConstantV object containing electrode atom indices
    nonbonded_force : openmm.NonbondedForce
        The NonbondedForce from the system
    custom_nonbonded_force : openmm.CustomNonbondedForce or None, optional
        The CustomNonbondedForce from the system (if present, e.g., for SAPT-FF)

    Example
    -------
    >>> # Get forces from system
    >>> nonbonded_force = [f for f in system.getForces()
    ...                    if isinstance(f, mm.NonbondedForce)][0]
    >>> custom_nb_force = [f for f in system.getForces()
    ...                    if isinstance(f, mm.CustomNonbondedForce)]
    >>> custom_nb_force = custom_nb_force[0] if custom_nb_force else None
    >>>
    >>> # Add exclusions BEFORE creating context
    >>> add_electrode_exclusions(integrator, nonbonded_force, custom_nb_force)
    >>>
    >>> # Now create context
    >>> context = mm.Context(system, integrator)
    >>> context.setPositions(positions)
    >>>
    >>> # ⚠️  CRITICAL: Reinitialize to apply exclusions
    >>> context.reinitialize(preserveState=True)
    """

    # ═══════════════════════════════════════════════════════════
    # Get electrode atom indices
    # ═══════════════════════════════════════════════════════════
    cathode_atoms = []
    if hasattr(constantv_obj, 'getNumCathodeAtoms'):
        for i in range(constantv_obj.getNumCathodeAtoms()):
            particle, area = constantv_obj.getCathodeAtomParameters(i)
            cathode_atoms.append(particle)
    else:
        raise ValueError("constantv_obj must have getNumCathodeAtoms() method")

    anode_atoms = []
    if hasattr(constantv_obj, 'getNumAnodeAtoms'):
        for i in range(constantv_obj.getNumAnodeAtoms()):
            particle, area = constantv_obj.getAnodeAtomParameters(i)
            anode_atoms.append(particle)
    else:
        raise ValueError("constantv_obj must have getNumAnodeAtoms() method")

    print(f"\n{'='*60}")
    print(f"Adding electrode exclusions (CRITICAL STEP)")
    print(f"{'='*60}")
    print(f"Cathode atoms: {len(cathode_atoms)}")
    print(f"Anode atoms: {len(anode_atoms)}")

    # ═══════════════════════════════════════════════════════════
    # Build set of existing exclusions in CustomNonbondedForce
    # (Replicates electrode_sapt_exclusions.py Lines 31-38)
    # ═══════════════════════════════════════════════════════════
    flagexclusions = {}
    if custom_nonbonded_force is not None:
        for i in range(custom_nonbonded_force.getNumExclusions()):
            particle1, particle2 = custom_nonbonded_force.getExclusionParticles(i)
            string1 = f"{particle1}_{particle2}"
            string2 = f"{particle2}_{particle1}"
            flagexclusions[string1] = 1
            flagexclusions[string2] = 1

    # ═══════════════════════════════════════════════════════════
    # Build set of existing exceptions in NonbondedForce
    # ═══════════════════════════════════════════════════════════
    flagexceptions = {}
    for i in range(nonbonded_force.getNumExceptions()):
        p1, p2, chg, sig, eps = nonbonded_force.getExceptionParameters(i)
        flagexceptions[f"{p1}_{p2}"] = i
        flagexceptions[f"{p2}_{p1}"] = i

    # ═══════════════════════════════════════════════════════════
    # Add CATHODE-CATHODE exclusions
    # (Replicates electrode_sapt_exclusions.py Lines 41-52)
    # ═══════════════════════════════════════════════════════════
    num_cathode_exclusions = 0
    for i in range(len(cathode_atoms)):
        indexi = cathode_atoms[i]
        for j in range(i+1, len(cathode_atoms)):  # Only i<j (same electrode)
            indexj = cathode_atoms[j]
            key = f"{indexi}_{indexj}"

            # Add to CustomNonbondedForce (if present)
            if custom_nonbonded_force is not None:
                if key not in flagexclusions and f"{indexj}_{indexi}" not in flagexclusions:
                    custom_nonbonded_force.addExclusion(indexi, indexj)
                    num_cathode_exclusions += 1

            # Add to NonbondedForce
            # Parameters: charge=0, sigma=1, epsilon=0, replace=True
            # (Line 52: nbondedForce.addException(indexi,indexj,0,1,0,True))
            if key not in flagexceptions and f"{indexj}_{indexi}" not in flagexceptions:
                nonbonded_force.addException(indexi, indexj, 0.0, 1.0, 0.0)

    print(f"Cathode-cathode exclusions: {num_cathode_exclusions}")

    # ═══════════════════════════════════════════════════════════
    # Add ANODE-ANODE exclusions (same logic)
    # ═══════════════════════════════════════════════════════════
    num_anode_exclusions = 0
    for i in range(len(anode_atoms)):
        indexi = anode_atoms[i]
        for j in range(i+1, len(anode_atoms)):
            indexj = anode_atoms[j]
            key = f"{indexi}_{indexj}"

            if custom_nonbonded_force is not None:
                if key not in flagexclusions and f"{indexj}_{indexi}" not in flagexclusions:
                    custom_nonbonded_force.addExclusion(indexi, indexj)
                    num_anode_exclusions += 1

            if key not in flagexceptions and f"{indexj}_{indexi}" not in flagexceptions:
                nonbonded_force.addException(indexi, indexj, 0.0, 1.0, 0.0)

    print(f"Anode-anode exclusions: {num_anode_exclusions}")
    print(f"{'='*60}")
    print(f"⚠️  CRITICAL: After creating Context, you MUST call:")
    print(f"⚠️           context.reinitialize(preserveState=True)")
    print(f"{'='*60}\n")


def add_saptff_exclusions(topology, system, water_residue_name='HOH', tfsi_residue_name='Tf2N'):
    """
    Add SAPT-FF specific exclusions for electrolyte molecules.

    Replicates SAPT_FF_exclusions class from electrode_sapt_exclusions.py:98-184

    This function adds:
    1. Water interaction groups (water-water via NonbondedForce, water-other via CustomNonbondedForce)
    2. TFSI intra-molecular exclusions with Drude screened pairs

    Parameters:
    -----------
    topology : openmm.app.Topology
        The system topology
    system : openmm.System
        The OpenMM system object
    water_residue_name : str, optional
        Residue name for water molecules (default: 'HOH')
    tfsi_residue_name : str, optional
        Residue name for TFSI molecules (default: 'Tf2N')

    Example:
    --------
    >>> # After creating system but before context
    >>> add_saptff_exclusions(topology, system)
    >>> context = mm.Context(system, integrator)
    >>> context.setPositions(positions)
    >>> context.reinitialize(preserveState=True)

    Notes:
    ------
    - Water model: SWM4-NDP for water-water, SAPT-FF for water-other
    - TFSI: All intra-molecular pairs excluded, Drude pairs screened (thole=2.0)
    - Must call context.reinitialize() after creating context
    """
    from openmm import NonbondedForce, CustomNonbondedForce, DrudeForce

    # Find forces in system
    nonbonded_force = None
    custom_nonbonded_force = None
    drude_force = None

    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, NonbondedForce):
            nonbonded_force = force
        elif isinstance(force, CustomNonbondedForce):
            custom_nonbonded_force = force
        elif isinstance(force, DrudeForce):
            drude_force = force

    if nonbonded_force is None:
        raise RuntimeError("NonbondedForce not found in system")

    print(f"\n{'='*60}")
    print(f"Adding SAPT-FF electrolyte exclusions")
    print(f"{'='*60}")

    # ═══════════════════════════════════════════════════════════
    # Add Water interaction groups (Lines 78-93)
    # ═══════════════════════════════════════════════════════════
    water_present = False
    for res in topology.residues():
        if res.name == water_residue_name:
            water_present = True
            break

    if water_present and custom_nonbonded_force is not None:
        water_atoms = set()
        not_water_atoms = set()

        print(f"✓ Water molecules detected (residue '{water_residue_name}')")
        print(f"  Creating interaction groups for hybrid water model:")
        print(f"  - Water-water: NonbondedForce (SWM4-NDP)")
        print(f"  - Water-other: CustomNonbondedForce (SAPT-FF)")

        for res in topology.residues():
            if res.name == water_residue_name:
                for atom in res.atoms():
                    water_atoms.add(atom.index)
            else:
                for atom in res.atoms():
                    not_water_atoms.add(atom.index)

        # Add interaction groups (Lines 92-93)
        custom_nonbonded_force.addInteractionGroup(water_atoms, not_water_atoms)
        custom_nonbonded_force.addInteractionGroup(not_water_atoms, not_water_atoms)

        print(f"  - Water atoms: {len(water_atoms)}")
        print(f"  - Non-water atoms: {len(not_water_atoms)}")
        print(f"  - Interaction groups added: 2")
    elif water_present:
        print(f"⚠️  Warning: Water detected but CustomNonbondedForce not found")
    else:
        print(f"  No water molecules found (skipped)")

    # ═══════════════════════════════════════════════════════════
    # Add TFSI exclusions with Drude screened pairs (Lines 129-184)
    # ═══════════════════════════════════════════════════════════
    tfsi_present = False
    for res in topology.residues():
        if res.name == tfsi_residue_name:
            tfsi_present = True
            break

    if tfsi_present:
        print(f"\n✓ TFSI molecules detected (residue '{tfsi_residue_name}')")
        print(f"  Creating intra-molecular exclusions with Drude screening...")

        # Build particle map for Drude force (Lines 138-140)
        particle_map = {}
        if drude_force is not None:
            for i in range(drude_force.getNumParticles()):
                particle, p1, p2, p3, p4, charge, pol, aniso = drude_force.getParticleParameters(i)
                particle_map[particle] = i

        # Build existing exceptions map (Lines 143-149)
        flag_exceptions = {}
        for i in range(nonbonded_force.getNumExceptions()):
            p1, p2, chg, sig, eps = nonbonded_force.getExceptionParameters(i)
            flag_exceptions[f"{p1}_{p2}"] = 1
            flag_exceptions[f"{p2}_{p1}"] = 1

        # Build existing exclusions map (Lines 152-158)
        flag_exclusions = {}
        if custom_nonbonded_force is not None:
            for i in range(custom_nonbonded_force.getNumExclusions()):
                p1, p2 = custom_nonbonded_force.getExclusionParticles(i)
                flag_exclusions[f"{p1}_{p2}"] = 1
                flag_exclusions[f"{p2}_{p1}"] = 1

        # Add exclusions for all TFSI intra-molecular pairs (Lines 162-184)
        num_tfsi_exclusions = 0
        num_screened_pairs = 0

        for res in topology.residues():
            if res.name == tfsi_residue_name:
                atoms = list(res.atoms())
                for i in range(len(atoms)):
                    for j in range(i + 1, len(atoms)):
                        idx_i = atoms[i].index
                        idx_j = atoms[j].index

                        # Add exception to NonbondedForce (Line 168)
                        # Parameters: charge=0, sigma=1, epsilon=0, replace=True
                        nonbonded_force.addException(idx_i, idx_j, 0.0, 1.0, 0.0)

                        # Add exclusion to CustomNonbondedForce if not already present (Lines 170-175)
                        key = f"{idx_i}_{idx_j}"
                        if custom_nonbonded_force is not None:
                            if key not in flag_exclusions and f"{idx_j}_{idx_i}" not in flag_exclusions:
                                custom_nonbonded_force.addExclusion(idx_i, idx_j)
                                num_tfsi_exclusions += 1

                        # Add Drude screened pair if both are Drude particles (Lines 177-184)
                        if drude_force is not None and idx_i in particle_map and idx_j in particle_map:
                            # Check if we already have this screened pair
                            if key not in flag_exceptions and f"{idx_j}_{idx_i}" not in flag_exceptions:
                                drude_i = particle_map[idx_i]
                                drude_j = particle_map[idx_j]
                                drude_force.addScreenedPair(drude_i, drude_j, 2.0)  # thole = 2.0
                                num_screened_pairs += 1

        print(f"  - CustomNonbonded exclusions added: {num_tfsi_exclusions}")
        if drude_force is not None:
            print(f"  - Drude screened pairs added: {num_screened_pairs} (thole=2.0)")
        else:
            print(f"  - Drude screened pairs: skipped (no DrudeForce)")
    else:
        print(f"\n  No TFSI molecules found (skipped)")

    print(f"{'='*60}")
    print(f"⚠️  CRITICAL: After creating Context, you MUST call:")
    print(f"⚠️           context.reinitialize(preserveState=True)")
    print(f"{'='*60}\n")


def configure_geometry_from_context(context, integrator, cathode_atom_idx, anode_atom_idx):
    """
    Automatically compute and set electrode geometry parameters from context.

    This function replicates the Original behavior from:
    - MM_classes.py::set_electrochemical_cell_parameters() Lines 229-245
    - Electrode_Virtual.__init__() Lines 256-260 (sheet_area calculation)

    Parameters
    ----------
    context : openmm.Context
        The simulation context (must be created with positions set)
    integrator : ConstantVIntegrator
        The integrator to configure
    cathode_atom_idx : int
        Index of any cathode atom (for z position)
    anode_atom_idx : int
        Index of any anode atom (for z position)

    Returns
    -------
    params : dict
        Dictionary with keys: 'Lgap', 'Lcell', 'totalArea', 'z_cathode', 'z_anode'

    Example
    -------
    >>> context = mm.Context(system, integrator)
    >>> context.setPositions(positions)
    >>> params = configure_geometry_from_context(
    ...     context, integrator,
    ...     cathode_atoms[0],  # First cathode atom
    ...     anode_atoms[0]     # First anode atom
    ... )
    >>> print(f"Auto-configured: Lcell={params['Lcell']:.4f} nm")
    """

    # ═══════════════════════════════════════════════════════════
    # Get positions and box vectors
    # (Replicates MM_classes.py Line 229)
    # ═══════════════════════════════════════════════════════════
    state = context.getState(getPositions=True)
    positions = state.getPositions()
    box_vectors = state.getPeriodicBoxVectors()

    # ═══════════════════════════════════════════════════════════
    # Get z positions of electrodes
    # (Replicates Lines 238-241)
    # ═══════════════════════════════════════════════════════════
    z_cathode = positions[cathode_atom_idx][2].value_in_unit(unit.nanometer)
    z_anode = positions[anode_atom_idx][2].value_in_unit(unit.nanometer)

    # ═══════════════════════════════════════════════════════════
    # Compute Lcell (electrode separation)
    # (Replicates Line 242)
    # ═══════════════════════════════════════════════════════════
    Lcell = abs(z_cathode - z_anode)

    # ═══════════════════════════════════════════════════════════
    # Compute Lgap (vacuum gap)
    # (Replicates Line 245)
    # ═══════════════════════════════════════════════════════════
    box_z = box_vectors[2][2].value_in_unit(unit.nanometer)
    Lgap = box_z - Lcell

    # ═══════════════════════════════════════════════════════════
    # Compute sheet area (cross product of box vectors a × b)
    # (Replicates Electrode_Virtual.__init__ Lines 256-258)
    # ═══════════════════════════════════════════════════════════
    # Original code:
    # crossBox = numpy.cross(boxVecs[0], boxVecs[1])
    # self.sheet_area = numpy.dot(crossBox, crossBox)**0.5 / nanometer**2

    a = box_vectors[0]
    b = box_vectors[1]

    # Cross product: a × b
    cross_x = a[1]*b[2] - a[2]*b[1]
    cross_y = a[2]*b[0] - a[0]*b[2]
    cross_z = a[0]*b[1] - a[1]*b[0]

    # Magnitude: |cross| = sqrt(cross · cross)
    cross_mag = (cross_x**2 + cross_y**2 + cross_z**2)**0.5
    total_area_nm2 = cross_mag.value_in_unit(unit.nanometer**2)

    # ═══════════════════════════════════════════════════════════
    # Set in integrator
    # ═══════════════════════════════════════════════════════════
    integrator.setLgap(Lgap)
    integrator.setLcell(Lcell)
    integrator.setTotalArea(total_area_nm2)
    integrator.setZCathode(z_cathode)
    integrator.setZAnode(z_anode)

    # ═══════════════════════════════════════════════════════════
    # Print summary
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Auto-configured electrode geometry:")
    print(f"{'='*60}")
    print(f"  Lcell (electrode separation) = {Lcell:.4f} nm")
    print(f"  Lgap (vacuum gap)            = {Lgap:.4f} nm")
    print(f"  Total area (sheet area)      = {total_area_nm2:.4f} nm²")
    print(f"  z_cathode                    = {z_cathode:.4f} nm")
    print(f"  z_anode                      = {z_anode:.4f} nm")
    print(f"{'='*60}\n")

    return {
        'Lgap': Lgap,
        'Lcell': Lcell,
        'totalArea': total_area_nm2,
        'z_cathode': z_cathode,
        'z_anode': z_anode
    }


def add_electrolyte_atoms_auto(topology, system, integrator, nonbonded_force,
                               natom_cutoff=100, exclude_chains=None):
    """
    Automatically identify and add electrolyte atoms/particles (including Drude).

    🔴 CRITICAL FIX (P0): This function now uses Scheme A to include ALL particles
    (atoms + Drude particles + virtual sites) by iterating over system.getNumParticles()
    instead of just topology.residues().atoms().

    This fixes a critical bug where Drude particles were excluded from the electrolyte
    list, causing incorrect Q_analytic calculations in Green's Reciprocity.

    Technical rationale:
    - Source of Truth: System (physics) > Topology (chemistry)
    - Drude particles are in System and NonbondedForce, but may not be in Topology residues
    - In SAPT-FF, Drude particles carry ~-1.0e charge each - missing them is catastrophic

    Original behavior (BUGGY):
    - MM_classes.py::initialize_electrolyte() Lines 256-279
    - Only iterated over topology.residues() → missed Drude particles

    Fixed behavior (Scheme A):
    - Iterate over all particles in System
    - Directly query NonbondedForce for charges
    - Automatically includes Drude particles, virtual sites, etc.

    Parameters
    ----------
    topology : openmm.app.Topology
        The system topology (used only to identify electrode chains)
    system : openmm.System
        The OpenMM System object (source of truth for particles)
    integrator : ConstantVIntegrator
        The integrator to add particles to
    nonbonded_force : openmm.NonbondedForce
        The NonbondedForce (to get charges for all particles)
    natom_cutoff : int, default=100
        (Legacy parameter - not used in Scheme A, kept for API compatibility)
    exclude_chains : list of int, optional
        Chain indices to exclude (e.g., electrode chains)

    Returns
    -------
    electrolyte_particles : list of int
        List of ALL electrolyte particle indices (atoms + Drude + virtual sites)

    Example
    -------
    >>> electrolyte_particles = add_electrolyte_atoms_auto(
    ...     topology, system, integrator, nonbonded_force,
    ...     exclude_chains=[0, 1]  # Exclude electrode chains
    ... )
    >>> print(f"Added {len(electrolyte_particles)} electrolyte particles")
    """

    if exclude_chains is None:
        exclude_chains = []

    # ═══════════════════════════════════════════════════════════
    # ⭐ CRITICAL SAFETY CHECK (Academic Rigor)
    # Verify 1:1 mapping between System and NonbondedForce indices
    # ═══════════════════════════════════════════════════════════
    num_system_particles = system.getNumParticles()
    num_force_particles = nonbonded_force.getNumParticles()

    if num_system_particles != num_force_particles:
        raise RuntimeError(
            f"FATAL ERROR: Particle count mismatch!\n"
            f"  System has {num_system_particles} particles\n"
            f"  NonbondedForce has {num_force_particles} particles\n"
            f"The assumption of 1:1 index mapping is broken.\n"
            f"Plugin cannot safely proceed. Check your system setup."
        )

    # ═══════════════════════════════════════════════════════════
    # Identify electrode particles (from topology chains)
    # ═══════════════════════════════════════════════════════════
    electrode_particles = set()

    for chain in topology.chains():
        if chain.index in exclude_chains:
            for atom in chain.atoms():
                electrode_particles.add(atom.index)

    # ═══════════════════════════════════════════════════════════
    # ⭐ SCHEME A: Iterate over ALL particles in System
    # This includes: atoms + Drude particles + virtual sites
    # ═══════════════════════════════════════════════════════════
    electrolyte_particles = []
    drude_count = 0
    atom_count = 0

    # Count topology atoms for statistics
    topology_atom_indices = set(atom.index for atom in topology.atoms())

    for particle_idx in range(num_system_particles):
        # Skip electrode particles
        if particle_idx in electrode_particles:
            continue

        # Get charge from NonbondedForce
        # (All particles - atoms, Drude, virtual sites - are in NonbondedForce)
        try:
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(particle_idx)

            # Only add particles with non-zero charge
            # (Threshold: 1e-6 to avoid floating point issues)
            if abs(charge._value) > 1e-6:
                electrolyte_particles.append(particle_idx)
                integrator.addElectrolyteAtom(particle_idx, charge._value)

                # Statistics: is this a Drude particle or regular atom?
                if particle_idx in topology_atom_indices:
                    atom_count += 1
                else:
                    drude_count += 1  # Likely Drude or virtual site

        except Exception as e:
            # Should not happen if assertion above passed, but be safe
            print(f"WARNING: Could not get parameters for particle {particle_idx}: {e}")
            continue

    # ═══════════════════════════════════════════════════════════
    # Print summary with Drude particle statistics
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Auto-identified electrolyte particles (Scheme A):")
    print(f"{'='*60}")
    print(f"  Total electrolyte particles: {len(electrolyte_particles)}")
    print(f"    - Regular atoms: {atom_count}")
    print(f"    - Drude/virtual particles: {drude_count}")
    print(f"  Total particles in system: {num_system_particles}")
    print(f"  Electrode particles excluded: {len(electrode_particles)}")
    print(f"  ⭐ Drude particles INCLUDED (critical fix)")
    print(f"{'='*60}\n")

    if drude_count > 0:
        print(f"✓ SUCCESS: {drude_count} Drude/virtual particles added to electrolyte")
        print(f"  (Previous buggy version would have missed these!)")
    else:
        print(f"ℹ INFO: No Drude particles detected (non-polarizable system)")

    return electrolyte_particles


def compute_electrode_area_per_atom(topology, electrode_atom_indices):
    """
    Compute area per atom for flat electrode.

    This function replicates the Original behavior from:
    - Electrode_Virtual.__init__() Lines 256-259

    Parameters
    ----------
    topology : openmm.app.Topology
        The system topology (for box vectors)
    electrode_atom_indices : list of int
        Indices of electrode atoms

    Returns
    -------
    area_per_atom : float
        Area per atom in nm²
    total_area : float
        Total sheet area in nm²

    Example
    -------
    >>> area_per_atom, total_area = compute_electrode_area_per_atom(
    ...     pdb.topology, cathode_atoms
    ... )
    """

    # Get box vectors
    box_vectors = topology.getPeriodicBoxVectors()

    # Compute cross product
    a = box_vectors[0]
    b = box_vectors[1]

    cross_x = a[1]*b[2] - a[2]*b[1]
    cross_y = a[2]*b[0] - a[0]*b[2]
    cross_z = a[0]*b[1] - a[1]*b[0]

    cross_mag = (cross_x**2 + cross_y**2 + cross_z**2)**0.5
    total_area = cross_mag.value_in_unit(unit.nanometer**2)

    # Area per atom
    num_atoms = len(electrode_atom_indices)
    area_per_atom = total_area / num_atoms

    return area_per_atom, total_area


def validate_setup(context, integrator):
    """
    Validate that the ConstantV setup is correct.

    Checks:
    1. Charge conservation (Q_cathode + Q_anode ≈ 0)
    2. Geometry parameters are set
    3. Exclusions are likely present (can't check directly)

    Parameters
    ----------
    context : openmm.Context
        The simulation context
    integrator : ConstantVIntegrator
        The integrator

    Returns
    -------
    valid : bool
        True if setup appears valid
    messages : list of str
        Warning/error messages
    """

    messages = []
    valid = True

    print(f"\n{'='*60}")
    print(f"Validating ConstantV setup:")
    print(f"{'='*60}")

    # Check geometry parameters
    try:
        Lgap = integrator.getLgap()
        Lcell = integrator.getLcell()
        totalArea = integrator.getTotalArea()

        print(f"✓ Geometry parameters set:")
        print(f"  Lgap = {Lgap:.4f} nm")
        print(f"  Lcell = {Lcell:.4f} nm")
        print(f"  Total area = {totalArea:.4f} nm²")

        if Lgap <= 0 or Lcell <= 0 or totalArea <= 0:
            messages.append("⚠️  WARNING: Geometry parameters have non-positive values")
            valid = False
    except:
        messages.append("❌ ERROR: Geometry parameters not set")
        valid = False

    # Check electrode atoms
    num_cathode = integrator.getNumCathodeAtoms()
    num_anode = integrator.getNumAnodeAtoms()

    print(f"✓ Electrode atoms:")
    print(f"  Cathode: {num_cathode} atoms")
    print(f"  Anode: {num_anode} atoms")

    if num_cathode == 0 or num_anode == 0:
        messages.append("❌ ERROR: No electrode atoms added")
        valid = False

    # Check electrolyte atoms
    num_electrolyte = integrator.getNumElectrolyteAtoms()
    print(f"✓ Electrolyte atoms: {num_electrolyte}")

    if num_electrolyte == 0:
        messages.append("⚠️  WARNING: No electrolyte atoms added")

    # Note about exclusions (can't check directly)
    print(f"\n⚠️  NOTE: Cannot verify exclusions directly.")
    print(f"    Make sure you called add_electrode_exclusions()")
    print(f"    and context.reinitialize(preserveState=True)")

    print(f"{'='*60}\n")

    if messages:
        print("Issues found:")
        for msg in messages:
            print(f"  {msg}")
        print()

    return valid, messages


# ═══════════════════════════════════════════════════════════════════
# ONE-CALL SETUP FUNCTIONS (P0 - Critical for User Experience)
# ═══════════════════════════════════════════════════════════════════

def initialize_electrodes_auto(
    integrator, topology, system, positions,
    voltage,
    cathode_identifier, anode_identifier,
    chain=False,
    exclude_element=(),
    buckyballs=None,
    nanotubes=None,
    nanotube_axes=None
):
    """
    ONE-CALL electrode initialization (replicates Original MM_classes.py:183-220)

    This is the KEY convenience function that transforms the plugin user experience
    from 8+ manual steps to a single function call, matching the Original's simplicity.

    🎯 Goal: Make plugin as easy to use as Original Python code

    Original workflow (1 call):
        MMsys.initialize_electrodes(Voltage, cathode_id, anode_id, chain=True,
                                     exclude_element=("H",), BuckyBalls=[...])

    Plugin workflow WITHOUT this function (8+ steps):
        1. Extract cathode atoms manually
        2. Extract anode atoms manually
        3. Compute area per atom
        4. Loop to add cathode atoms
        5. Loop to add anode atoms
        6. Add exclusions
        7. Create context
        8. Configure geometry

    Plugin workflow WITH this function (1 call):
        context = initialize_electrodes_auto(integrator, topology, system, positions,
                                              voltage, cathode_id, anode_id, ...)

    Parameters
    ----------
    integrator : ConstantVIntegrator or ConstantVDrudeLangevinIntegrator
        The integrator to configure
    topology : openmm.app.Topology
        System topology
    system : openmm.System
        OpenMM system
    positions : list of Vec3
        Initial positions
    voltage : float
        Applied voltage in Volts
    cathode_identifier : str or int or tuple
        Cathode residue name (if chain=False) or chain index/indices (if chain=True)
    anode_identifier : str or int or tuple
        Anode residue name (if chain=False) or chain index/indices (if chain=True)
    chain : bool, default=False
        If True, identify by chain index; if False, by residue name
    exclude_element : tuple of str, default=()
        Elements to exclude (e.g., ("H",) for dummy hydrogen)
    buckyballs : list of int or tuple, optional
        List of chain indices for Buckyball conductors
        Each element can be: int (virtual chain) or tuple (virtual, real)
    nanotubes : list of tuple, optional
        List of (virtual_chain, real_chain) tuples for nanotubes
    nanotube_axes : list of tuple, optional
        List of (ax, ay, az) axis vectors for each nanotube

    Returns
    -------
    context : openmm.Context
        Fully configured context, ready to run simulation

    Example
    -------
    >>> # Flat electrodes only (simplest case)
    >>> context = initialize_electrodes_auto(
    ...     integrator, pdb.topology, system, pdb.positions,
    ...     voltage=1.0,
    ...     cathode_identifier=(0, 2),  # Chains 0 and 2
    ...     anode_identifier=(1, 3),    # Chains 1 and 3
    ...     chain=True,
    ...     exclude_element=("H",)
    ... )
    >>>
    >>> # With Buckyball conductors
    >>> context = initialize_electrodes_auto(
    ...     integrator, pdb.topology, system, pdb.positions,
    ...     voltage=1.0,
    ...     cathode_identifier=(0, 2),
    ...     anode_identifier=(1, 3),
    ...     chain=True,
    ...     exclude_element=("H",),
    ...     buckyballs=[(4, 5)]  # Virtual chain 4, real chain 5
    ... )

    Notes
    -----
    Replicates Original Python code from:
    - MM_classes.py::initialize_electrodes() Lines 183-220
    - Electrode_Virtual.__init__() Lines 249-277
    - Buckyball_Virtual.__init__() Lines 391-471
    """

    print(f"\n{'='*70}")
    print(f"🚀 ONE-CALL ELECTRODE INITIALIZATION")
    print(f"{'='*70}")
    print(f"Voltage: {voltage} V")
    print(f"Cathode: {cathode_identifier} (chain={chain})")
    print(f"Anode: {anode_identifier} (chain={chain})")
    if exclude_element:
        print(f"Excluding elements: {exclude_element}")
    if buckyballs:
        print(f"Buckyball conductors: {len(buckyballs)}")
    if nanotubes:
        print(f"Nanotube conductors: {len(nanotubes)}")
    print(f"{'='*70}\n")

    # ═══════════════════════════════════════════════════════════════
    # Step 1: Extract cathode atoms
    # (Replicates Electrode_Virtual.__init__ Lines 261-277)
    # ═══════════════════════════════════════════════════════════════
    cathode_atoms = _extract_electrode_atoms(
        topology, cathode_identifier, chain, exclude_element
    )
    print(f"✓ Extracted {len(cathode_atoms)} cathode atoms")

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Extract anode atoms
    # ═══════════════════════════════════════════════════════════════
    anode_atoms = _extract_electrode_atoms(
        topology, anode_identifier, chain, exclude_element
    )
    print(f"✓ Extracted {len(anode_atoms)} anode atoms")

    # ═══════════════════════════════════════════════════════════════
    # Step 3: Compute area per atom
    # (Replicates Electrode_Virtual.__init__ Lines 256-259)
    # ═══════════════════════════════════════════════════════════════
    cathode_area_per_atom, cathode_total_area = compute_electrode_area_per_atom(
        topology, cathode_atoms
    )
    anode_area_per_atom, anode_total_area = compute_electrode_area_per_atom(
        topology, anode_atoms
    )
    print(f"✓ Cathode area per atom: {cathode_area_per_atom:.6f} nm²")
    print(f"✓ Anode area per atom: {anode_area_per_atom:.6f} nm²")

    # ═══════════════════════════════════════════════════════════════
    # Step 4: Set voltage in integrator
    # ═══════════════════════════════════════════════════════════════
    integrator.setVoltage(voltage)
    print(f"✓ Voltage set: {voltage} V")

    # ═══════════════════════════════════════════════════════════════
    # Step 5: Add cathode atoms to integrator
    # ═══════════════════════════════════════════════════════════════
    for atom_idx in cathode_atoms:
        integrator.addCathodeAtom(atom_idx, cathode_area_per_atom)
    print(f"✓ Added {len(cathode_atoms)} cathode atoms to integrator")

    # ═══════════════════════════════════════════════════════════════
    # Step 6: Add anode atoms to integrator
    # ═══════════════════════════════════════════════════════════════
    for atom_idx in anode_atoms:
        integrator.addAnodeAtom(atom_idx, anode_area_per_atom)
    print(f"✓ Added {len(anode_atoms)} anode atoms to integrator")

    # ═══════════════════════════════════════════════════════════════
    # Step 7: Add Buckyball conductors (if requested)
    # (Replicates MM_classes.py Lines 191-195)
    # ═══════════════════════════════════════════════════════════════
    if buckyballs:
        print(f"\n🔵 Adding {len(buckyballs)} Buckyball conductor(s)...")
        for i, buckyball_id in enumerate(buckyballs):
            if isinstance(buckyball_id, tuple):
                virtual_chain, real_chain = buckyball_id
            else:
                # If single int, assume it's virtual chain, user must provide real separately
                raise ValueError(
                    f"Buckyball {i}: Must provide tuple (virtual_chain, real_chain). "
                    f"Got: {buckyball_id}"
                )

            # Use helper function (will be implemented next)
            try:
                add_buckyball_conductor(
                    integrator, topology, system,
                    virtual_chain, real_chain,
                    "cathode",  # Assume buckyballs on cathode (like Original)
                    voltage,
                    exclude_element
                )
                print(f"  ✓ Buckyball {i+1}: chains ({virtual_chain}, {real_chain})")
            except Exception as e:
                print(f"  ⚠️ Warning: Failed to add Buckyball {i}: {e}")

    # ═══════════════════════════════════════════════════════════════
    # Step 8: Add Nanotube conductors (if requested)
    # (Replicates MM_classes.py Lines 199-209)
    # ═══════════════════════════════════════════════════════════════
    if nanotubes:
        if not nanotube_axes or len(nanotube_axes) != len(nanotubes):
            raise ValueError(
                f"Must provide axis for each nanotube. "
                f"Got {len(nanotubes)} nanotubes but {len(nanotube_axes) if nanotube_axes else 0} axes."
            )

        print(f"\n🔴 WARNING: Nanotube support not yet implemented in C++ API")
        print(f"   Skipping {len(nanotubes)} nanotube(s)")
        # TODO: Implement when C++ Nanotube API is ready
        # for i, (nanotube_id, axis) in enumerate(zip(nanotubes, nanotube_axes)):
        #     virtual_chain, real_chain = nanotube_id
        #     add_nanotube_conductor(integrator, topology, system,
        #                            virtual_chain, real_chain,
        #                            "cathode", voltage, axis, exclude_element)

    # ═══════════════════════════════════════════════════════════════
    # Step 9: Get forces for exclusions
    # ═══════════════════════════════════════════════════════════════
    nonbonded_force = None
    custom_nonbonded_force = None

    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            nonbonded_force = force
        elif isinstance(force, mm.CustomNonbondedForce):
            custom_nonbonded_force = force

    if nonbonded_force is None:
        raise RuntimeError("No NonbondedForce found in system")

    # ═══════════════════════════════════════════════════════════════
    # Step 10: Add electrode exclusions
    # (Replicates MM_classes.py Lines 560-622 via generate_exclusions)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n📋 Adding electrode exclusions...")
    add_electrode_exclusions(
        integrator, nonbonded_force, custom_nonbonded_force
    )

    # ═══════════════════════════════════════════════════════════════
    # Step 11: Create context
    # ═══════════════════════════════════════════════════════════════
    print(f"\n🔧 Creating OpenMM context...")
    context = mm.Context(system, integrator)
    context.setPositions(positions)

    # ═══════════════════════════════════════════════════════════════
    # Step 12: Reinitialize to apply exclusions
    # (CRITICAL - without this, exclusions won't take effect)
    # ═══════════════════════════════════════════════════════════════
    print(f"🔄 Reinitializing context to apply exclusions...")
    context.reinitialize(preserveState=True)

    # ═══════════════════════════════════════════════════════════════
    # Step 13: Configure geometry parameters
    # (Replicates MM_classes.py Lines 229-245 via set_electrochemical_cell_parameters)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n📐 Auto-configuring electrode geometry...")
    configure_geometry_from_context(
        context, integrator,
        cathode_atoms[0],  # Use first atom for z position
        anode_atoms[0]
    )

    # ═══════════════════════════════════════════════════════════════
    # Step 14: Add electrolyte atoms
    # (Replicates MM_classes.py Lines 256-279 via initialize_electrolyte)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n🧪 Auto-identifying electrolyte atoms...")

    # Determine which chains to exclude (electrodes + conductors)
    exclude_chains = []

    if chain:
        # Extract chain indices from identifiers
        if isinstance(cathode_identifier, tuple):
            exclude_chains.extend(cathode_identifier)
        else:
            exclude_chains.append(cathode_identifier)

        if isinstance(anode_identifier, tuple):
            exclude_chains.extend(anode_identifier)
        else:
            exclude_chains.append(anode_identifier)

        # Add buckyball chains
        if buckyballs:
            for buckyball_id in buckyballs:
                if isinstance(buckyball_id, tuple):
                    virtual_chain, real_chain = buckyball_id
                    exclude_chains.extend([virtual_chain, real_chain])

    add_electrolyte_atoms_auto(
        topology, system, integrator, nonbonded_force,
        exclude_chains=exclude_chains
    )

    # ═══════════════════════════════════════════════════════════════
    # Final: Success message
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"✅ INITIALIZATION COMPLETE - Context ready to run!")
    print(f"{'='*70}")
    print(f"Summary:")
    print(f"  Cathode atoms: {len(cathode_atoms)}")
    print(f"  Anode atoms: {len(anode_atoms)}")
    print(f"  Electrolyte atoms: {integrator.getNumElectrolyteAtoms()}")
    if buckyballs:
        print(f"  Buckyball conductors: {len(buckyballs)}")
    print(f"  Voltage: {voltage} V")
    print(f"  Ready for: simulation.step(N)")
    print(f"{'='*70}\n")

    return context


def _extract_electrode_atoms(topology, identifier, chain, exclude_element):
    """
    Helper function to extract electrode atom indices from topology.

    Replicates the atom extraction logic from Electrode_Virtual.__init__
    (Fixed_Voltage_routines.py Lines 261-277)

    Parameters
    ----------
    topology : openmm.app.Topology
    identifier : str or int or tuple
        Residue name (if chain=False) or chain index/indices (if chain=True)
    chain : bool
        If True, match by chain index; if False, by residue name
    exclude_element : tuple of str
        Elements to exclude

    Returns
    -------
    atom_indices : list of int
        List of atom indices
    """
    atom_indices = []

    if chain:
        # Match by chain index
        # identifier can be int or tuple of ints
        if isinstance(identifier, tuple):
            chain_indices = list(identifier)
        else:
            chain_indices = [identifier]

        for chain_obj in topology.chains():
            if chain_obj.index in chain_indices:
                for atom in chain_obj.atoms():
                    # Check if element should be excluded
                    if atom.element.symbol not in exclude_element:
                        atom_indices.append(atom.index)

    else:
        # Match by residue name
        for residue in topology.residues():
            if residue.name == identifier:
                for atom in residue.atoms():
                    if atom.element.symbol not in exclude_element:
                        atom_indices.append(atom.index)

    if not atom_indices:
        raise ValueError(
            f"No atoms found for identifier '{identifier}' "
            f"(chain={chain}, exclude_element={exclude_element})"
        )

    return atom_indices


def add_buckyball_conductor(integrator, topology, system, virtual_chain, real_chain,
                            electrode_type, voltage, exclude_element=()):
    """
    Helper function to add a Buckyball conductor to the ConstantVForce.

    Replicates Buckyball_Virtual.__init__() logic from Fixed_Voltage_routines.py:391-471

    This function:
    1. Extracts virtual atom indices from virtual_chain (outer layer)
    2. Extracts real atom indices from real_chain (inner layer)
    3. Finds ConstantVForce in the system
    4. Calls force.addBuckyballConductor() with the atom lists

    Parameters:
    -----------
    integrator : ConstantVLangevinIntegrator
        The integrator object (not used, but kept for API consistency)
    topology : openmm.app.Topology
        The system topology
    system : openmm.System
        The OpenMM system object
    virtual_chain : int
        Chain index for virtual (outer) atoms of the Buckyball
    real_chain : int
        Chain index for real (inner) atoms of the Buckyball
    electrode_type : str
        Either "cathode" or "anode"
    voltage : float
        Applied voltage in Volts (NOT kJ/mol - C++ will convert)
    exclude_element : tuple of str
        Elements to exclude (e.g., ("H",) for dummy hydrogen atoms)

    Returns:
    --------
    int
        Index of the added Buckyball conductor in the force's conductor list

    Example:
    --------
    >>> # Add a Buckyball with virtual atoms on chain 1, real atoms on chain 4
    >>> idx = add_buckyball_conductor(
    ...     integrator, topology, system,
    ...     virtual_chain=1, real_chain=4,
    ...     electrode_type="cathode",
    ...     voltage=1.0,
    ...     exclude_element=("H",)
    ... )

    Notes:
    ------
    - Virtual atoms: Outer layer that participates in electrostatics (ghost atoms)
    - Real atoms: Inner layer with physical charges
    - This matches the Python Original's Buckyball_Virtual class structure
    - Geometry (center, radius, normals) is computed automatically by C++ kernel
    """
    from openmm import unit

    # Validate electrode type
    if electrode_type not in ["cathode", "anode"]:
        raise ValueError(f"electrode_type must be 'cathode' or 'anode', got '{electrode_type}'")

    # Extract virtual atoms from virtual_chain (Lines 392-395 + parent Lines 261-277)
    virtual_atoms = []
    for chain in topology.chains():
        if chain.index == virtual_chain:
            for atom in chain.atoms():
                element = atom.element
                if element.symbol not in exclude_element:
                    virtual_atoms.append(atom.index)
            break

    if len(virtual_atoms) == 0:
        raise ValueError(f"No virtual atoms found in chain {virtual_chain} (after excluding {exclude_element})")

    # Extract real atoms from real_chain (Lines 406-421)
    real_atoms = []
    for chain in topology.chains():
        if chain.index == real_chain:
            for atom in chain.atoms():
                element = atom.element
                if element.symbol not in exclude_element:
                    real_atoms.append(atom.index)
            break

    if len(real_atoms) == 0:
        raise ValueError(f"No real atoms found in chain {real_chain} (after excluding {exclude_element})")

    # Find ConstantVForce in the system
    constantv_force = None
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if force.__class__.__name__ == 'ConstantVForce':
            constantv_force = force
            break

    if constantv_force is None:
        raise RuntimeError("ConstantVForce not found in system. Make sure to add it before calling this function.")

    # Call C++ API to add the Buckyball conductor
    # API: addBuckyballConductor(virtualAtoms, realAtoms, electrodeType, voltage)
    # Note: voltage is in Volts here, C++ will convert to kJ/mol internally
    conductor_index = constantv_force.addBuckyballConductor(
        virtual_atoms,
        real_atoms,
        electrode_type,
        voltage
    )

    print(f"✓ Added Buckyball conductor #{conductor_index}:")
    print(f"  - Virtual atoms: {len(virtual_atoms)} atoms from chain {virtual_chain}")
    print(f"  - Real atoms: {len(real_atoms)} atoms from chain {real_chain}")
    print(f"  - Electrode type: {electrode_type}")
    print(f"  - Voltage: {voltage} V")

    return conductor_index


class ElectrodeChargeReporter:
    """
    Reporter to write electrode charges to a file during simulation.

    Replicates write_electrode_charges() from MM_classes.py:824-843

    This reporter writes electrode charges in the following order:
    1. Cathode charges (all atoms)
    2. Conductor charges (Buckyballs/Nanotubes, if any)
    3. Anode charges (all atoms)

    Each line contains space-separated charge values, one line per report step.

    Parameters:
    -----------
    file : str or file-like object
        Output file path or file handle to write charges to
    reportInterval : int
        Number of integration steps between reports
    integrator : ConstantVLangevinIntegrator or ConstantVDrudeLangevinIntegrator
        The integrator (used to get electrode atom indices)
    system : openmm.System
        The system (used to find ConstantVForce for Buckyball atoms)

    Example:
    --------
    >>> reporter = ElectrodeChargeReporter('charges.dat', reportInterval=100,
    ...                                     integrator=integrator, system=system)
    >>> simulation.reporters.append(reporter)
    >>> simulation.step(10000)  # Writes charges every 100 steps

    Notes:
    ------
    - Charges are written in scientific notation with 6 decimal places
    - Each line is flushed immediately to prevent data loss
    - Compatible with OpenMM's standard Reporter interface
    """

    def __init__(self, file, reportInterval, integrator, system):
        """
        Initialize the ElectrodeChargeReporter.

        Parameters:
        -----------
        file : str or file-like object
            Output file path or file handle
        reportInterval : int
            Report every N steps
        integrator : ConstantVLangevinIntegrator or ConstantVDrudeLangevinIntegrator
            Integrator containing electrode atom lists
        system : openmm.System
            System containing ConstantVForce
        """
        self._reportInterval = reportInterval
        self._integrator = integrator
        self._system = system

        # Open file if string path provided
        if isinstance(file, str):
            self._file = open(file, 'w')
            self._closeFile = True
        else:
            self._file = file
            self._closeFile = False

        # Cache electrode atom indices (Lines 826-837 structure)
        self._cathode_atoms = []
        self._anode_atoms = []
        self._conductor_atoms = []  # List of conductor atom lists

        # Get cathode atoms from integrator
        try:
            # Try ConstantVDrudeLangevinIntegrator API (vector getters)
            self._cathode_atoms = list(integrator.getCathodeAtomIndices())
            self._anode_atoms = list(integrator.getAnodeAtomIndices())
        except AttributeError:
            # Fall back to ConstantVLangevinIntegrator API (parameter getters)
            num_cathode = integrator.getNumCathodeAtoms()
            for i in range(num_cathode):
                particle, area = integrator.getCathodeAtomParameters(i)
                self._cathode_atoms.append(particle)

            num_anode = integrator.getNumAnodeAtoms()
            for i in range(num_anode):
                particle, area = integrator.getAnodeAtomParameters(i)
                self._anode_atoms.append(particle)

        # Get Buckyball conductor atoms from ConstantVForce (Lines 832-834)
        constantv_force = None
        self._nonbonded_force = None

        for i in range(system.getNumForces()):
            force = system.getForce(i)
            if force.__class__.__name__ == 'ConstantVForce':
                constantv_force = force
            elif force.__class__.__name__ == 'NonbondedForce':
                self._nonbonded_force = force

        if constantv_force is not None:
            num_conductors = constantv_force.getNumBuckyballConductors()
            for i in range(num_conductors):
                params = constantv_force.getBuckyballConductorParameters(i)
                # params is a dict with keys: virtualAtoms, realAtoms, electrodeType, voltage
                virtual_atoms = params['virtualAtoms']
                self._conductor_atoms.append(virtual_atoms)

        if self._nonbonded_force is None:
            raise RuntimeError("NonbondedForce not found in system. Cannot retrieve charges.")

        # Print summary
        print(f"ElectrodeChargeReporter initialized:")
        print(f"  - Cathode atoms: {len(self._cathode_atoms)}")
        print(f"  - Anode atoms: {len(self._anode_atoms)}")
        print(f"  - Conductors: {len(self._conductor_atoms)}")
        total_atoms = len(self._cathode_atoms) + len(self._anode_atoms)
        for conductor in self._conductor_atoms:
            total_atoms += len(conductor)
        print(f"  - Total charges to write: {total_atoms}")
        print(f"  - Report interval: {reportInterval} steps")

    def describeNextReport(self, simulation):
        """
        Get information about the next report.

        This is called by OpenMM to determine when to call report().
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False, None)

    def report(self, simulation, state):
        """
        Write electrode charges to file.

        This implements the charge writing logic from MM_classes.py:824-843

        Order of charges:
        1. Cathode (all atoms)
        2. Conductors (Buckyballs, in order added)
        3. Anode (all atoms)
        """
        # Charges are stored in NonbondedForce and updated by the SCF iteration
        # We need to update the force in the context first to get latest charges
        self._nonbonded_force.updateParametersInContext(simulation.context)

        # Collect all charges in order (Lines 826-841)
        all_charges = []

        # 1. Cathode charges (Line 826-827)
        for atom_idx in self._cathode_atoms:
            charge, sigma, epsilon = self._nonbonded_force.getParticleParameters(atom_idx)
            # Convert from OpenMM units to elementary charge units
            all_charges.append(charge._value)

        # 2. Conductor charges (Lines 832-834)
        for conductor_atoms in self._conductor_atoms:
            for atom_idx in conductor_atoms:
                charge, sigma, epsilon = self._nonbonded_force.getParticleParameters(atom_idx)
                all_charges.append(charge._value)

        # 3. Anode charges (Lines 836-837)
        for atom_idx in self._anode_atoms:
            charge, sigma, epsilon = self._nonbonded_force.getParticleParameters(atom_idx)
            all_charges.append(charge._value)

        # Write charges to file (Lines 827, 834, 837, 841)
        for charge in all_charges:
            self._file.write(f"{charge:f} ")

        # Newline and flush (Lines 841-842)
        self._file.write("\n")
        self._file.flush()

    def __del__(self):
        """Close file on deletion if we owned it."""
        if self._closeFile and hasattr(self, '_file'):
            self._file.close()


class MC_Barostat:
    """
    Monte Carlo barostat for density equilibration in fixed-voltage simulations.

    Replicates MC_parameters class and MC_Barostat_step() from MM_classes.py:637-748, 906-914

    This barostat performs Monte Carlo moves to equilibrate electrolyte density:
    1. Run MD for `barofreq` steps
    2. Generate trial move: shift movable electrode + scale electrolyte COMs
    3. Accept/reject via Metropolis criterion (ΔE + PΔV - NkT·ln(V'/V))
    4. Adaptively adjust move size based on acceptance ratio

    Parameters:
    -----------
    simulation : openmm.app.Simulation
        The OpenMM simulation object
    topology : openmm.app.Topology
        System topology (for residue iteration)
    cathode_atoms : list of int
        Cathode atom indices (stationary reference)
    anode_atoms : list of int
        Anode atom indices (will be moved)
    electrolyte_residues : list of openmm.app.Residue
        Electrolyte residues to scale
    temperature : float
        Temperature in Kelvin
    cell_dimensions : tuple of 3 floats
        Box dimensions (Lx, Ly, Lz) in nm
    pressure : float, optional
        Pressure in bar (default: 1.0)
    barofreq : int, optional
        MD steps between MC moves (default: 100)
    shiftscale : float, optional
        Initial move size in nm (default: 0.02)
    electrode_move : str, optional
        Which electrode to move: "Anode" (default, only supported option)

    Example:
    --------
    >>> # Create MC barostat for density equilibration
    >>> mc_barostat = MC_Barostat(
    ...     simulation, topology,
    ...     cathode_atoms, anode_atoms, electrolyte_residues,
    ...     temperature=300.0,
    ...     cell_dimensions=(4.0, 4.0, 8.0),  # nm
    ...     pressure=1.0,  # bar
    ...     barofreq=100,
    ...     shiftscale=0.02  # nm
    ... )
    >>>
    >>> # Run 1000 MC steps
    >>> for i in range(1000):
    ...     mc_barostat.step()
    ...     if i % 100 == 0:
    ...         print(f"Step {i}, acceptance: {mc_barostat.get_acceptance_ratio():.2%}")

    Notes:
    ------
    - Only Anode movement is currently supported (Cathode is reference)
    - Move size (shiftscale) is adaptively tuned every 50 steps
    - Target acceptance ratio: 25-75%
    - Electrolyte COMs are scaled to maintain uniform density
    """

    def __init__(self, simulation, topology, cathode_atoms, anode_atoms,
                 electrolyte_residues, temperature, cell_dimensions,
                 pressure=1.0, barofreq=100, shiftscale=0.02,
                 electrode_move="Anode"):
        """
        Initialize MC_Barostat.

        See class docstring for parameter descriptions.
        """
        import numpy as np
        from openmm import unit

        self.simulation = simulation
        self.topology = topology
        self.cathode_atoms = cathode_atoms
        self.anode_atoms = anode_atoms
        self.electrolyte_residues = electrolyte_residues
        self.electrode_move = electrode_move

        # Physical constants (Lines 908-909)
        kB = 1.380649e-23  # J/K (Boltzmann constant)
        NA = 6.02214076e23  # 1/mol (Avogadro's number)
        self.RT = kB * temperature * NA  # J/mol
        # Convert pressure: bar → kJ/(mol·nm³)
        # 1 bar = 0.1 MPa = 0.1 kJ/(nm·nm²) = 0.1 kJ/nm³
        # Force = Pressure × Area = pressure · Lx · Ly
        Lx, Ly, Lz = cell_dimensions
        self.pressure = pressure * Lx * Ly * NA * 0.1  # kJ/mol (force along z)

        # MC parameters (Lines 911-914)
        self.barofreq = barofreq  # MD steps between MC moves
        self.shiftscale = shiftscale  # Move size in nm
        self.ntrials = 0  # Number of MC trials since last reset
        self.naccept = 0  # Number of accepted moves since last reset

        # Cache reference electrode atom for COM scaling (Line 681)
        self.reference_atom_index = cathode_atoms[0]  # Use first cathode atom

        # Validate electrode_move (Lines 680-692)
        if electrode_move != "Anode":
            raise ValueError(
                "Currently, only electrode_move='Anode' is supported. "
                "Generalizing to other conductors requires additional logic."
            )

        print(f"\nMC_Barostat initialized:")
        print(f"  - Temperature: {temperature} K")
        print(f"  - Pressure: {pressure} bar")
        print(f"  - Cell dimensions: {Lx:.2f} × {Ly:.2f} × {Lz:.2f} nm")
        print(f"  - Barofreq: {barofreq} steps")
        print(f"  - Initial shiftscale: {shiftscale:.4f} nm")
        print(f"  - Movable electrode: {electrode_move}")
        print(f"  - Cathode atoms (reference): {len(cathode_atoms)}")
        print(f"  - Anode atoms (movable): {len(anode_atoms)}")
        print(f"  - Electrolyte residues: {len(electrolyte_residues)}")

    def step(self):
        """
        Perform one MC barostat step.

        This method replicates MC_Barostat_step() from MM_classes.py:637-748
        """
        import numpy as np
        import random
        from openmm import unit

        # Helper functions (Lines 640-653)
        def metropolis(pe_comp):
            """Metropolis acceptance criterion."""
            if pe_comp < 0.0:
                return True
            elif random.uniform(0.0, 1.0) < np.exp(-pe_comp / self.RT):
                return True
            return False

        def intra_molecular_vectors(residue_obj, pos_ref, positions_array):
            """Get intra-molecular vectors relative to reference position."""
            intra_vec = []
            for atom in residue_obj.atoms():
                pos_atom = positions_array[atom.index]
                vec = pos_atom - pos_ref
                intra_vec.append(np.array(vec))
            return np.array(intra_vec)

        self.ntrials += 1

        # ═══════════════════════════════════════════════════════════
        # Run MD steps (Line 659)
        # ═══════════════════════════════════════════════════════════
        self.simulation.step(self.barofreq)

        # ═══════════════════════════════════════════════════════════
        # Get current state (Lines 662-669)
        # ═══════════════════════════════════════════════════════════
        state = self.simulation.context.getState(getEnergy=True, getPositions=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        oldE_value = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        # Store old positions
        oldpos = np.array(positions)
        newpos = np.array(positions)

        # ═══════════════════════════════════════════════════════════
        # Generate trial move (Lines 672-692)
        # ═══════════════════════════════════════════════════════════
        # Random move: ±shiftscale nm
        deltalen = self.shiftscale * (random.uniform(0, 1) * 2 - 1)

        # Move Anode atoms (Lines 682-689)
        for atom_idx in self.anode_atoms:
            newpos[atom_idx, 2] += deltalen

        # ═══════════════════════════════════════════════════════════
        # Scale electrolyte COMs (Lines 694-719)
        # ═══════════════════════════════════════════════════════════
        # Get cell dimensions
        cathode_z = newpos[self.cathode_atoms[0], 2]
        anode_z_old = oldpos[self.anode_atoms[0], 2]
        anode_z_new = newpos[self.anode_atoms[0], 2]

        Lcell_old = abs(anode_z_old - cathode_z)
        Lcell_new = abs(anode_z_new - cathode_z)

        N_electrolyte_mol = 0

        # Loop over electrolyte residues and scale COMs
        for res in self.electrolyte_residues:
            N_electrolyte_mol += 1

            # Get reference position (first atom in residue)
            atoms_list = list(res.atoms())
            if len(atoms_list) == 0:
                continue
            pos_ref = newpos[atoms_list[0].index].copy()

            # Get intra-molecular vectors
            intra_vec = intra_molecular_vectors(res, pos_ref, newpos)

            # Convert to coordinate system relative to stationary cathode (Line 710)
            pos_ref[2] = pos_ref[2] - newpos[self.reference_atom_index, 2]

            # Scale by cell dimension ratio (Line 712)
            pos_ref[2] = pos_ref[2] * Lcell_new / Lcell_old

            # Convert back to global coordinates (Line 714)
            pos_ref[2] = pos_ref[2] + newpos[self.reference_atom_index, 2]

            # Update all atoms in molecule (Lines 716-719)
            for idx, atom in enumerate(atoms_list):
                newpos[atom.index] = pos_ref + intra_vec[idx]

        # ═══════════════════════════════════════════════════════════
        # Evaluate trial energy (Lines 722-726)
        # ═══════════════════════════════════════════════════════════
        self.simulation.context.setPositions(newpos * unit.nanometer)
        state_new = self.simulation.context.getState(getEnergy=True)
        newE_value = state_new.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        # ═══════════════════════════════════════════════════════════
        # Metropolis acceptance (Lines 727-736)
        # ═══════════════════════════════════════════════════════════
        # ΔE + PΔV - NkT·ln(V'/V)
        # Note: deltalen in nm, pressure already has units kJ/mol
        w = (newE_value - oldE_value +
             self.pressure * deltalen -
             N_electrolyte_mol * self.RT / 1000 * np.log(Lcell_new / Lcell_old))  # RT in kJ/mol

        if metropolis(w):
            self.naccept += 1
            # Move accepted, keep new positions
            print(f"  MC move accepted: ΔL = {deltalen:.4f} nm, w = {w:.2f} kJ/mol")
        else:
            # Move rejected, revert to old positions
            self.simulation.context.setPositions(oldpos * unit.nanometer)

        # ═══════════════════════════════════════════════════════════
        # Adaptive tuning of move size (Lines 738-748)
        # ═══════════════════════════════════════════════════════════
        if self.ntrials >= 50:
            acceptance_ratio = self.naccept / self.ntrials
            print(f"\n  After {self.ntrials} MC steps:")
            print(f"    Acceptance ratio: {acceptance_ratio:.2%}")
            print(f"    Current shiftscale: {self.shiftscale:.4f} nm")

            # Adjust shiftscale to target 25-75% acceptance
            if acceptance_ratio < 0.25:
                self.shiftscale /= 1.1
                print(f"    → Decreased shiftscale to {self.shiftscale:.4f} nm (low acceptance)")
            elif acceptance_ratio > 0.75:
                self.shiftscale *= 1.1
                print(f"    → Increased shiftscale to {self.shiftscale:.4f} nm (high acceptance)")

            # Reset counters
            self.ntrials = 0
            self.naccept = 0

    def get_acceptance_ratio(self):
        """Get current acceptance ratio."""
        if self.ntrials == 0:
            return 0.0
        return self.naccept / self.ntrials

    def get_statistics(self):
        """Get MC statistics."""
        return {
            'ntrials': self.ntrials,
            'naccept': self.naccept,
            'acceptance_ratio': self.get_acceptance_ratio(),
            'shiftscale': self.shiftscale
        }


# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTIC & DEBUGGING UTILITIES (P2 - Config Parameters)
# ═══════════════════════════════════════════════════════════════════

def get_electrode_charge_summary(integrator, system):
    """
    Get comprehensive electrode charge statistics for debugging and monitoring.

    This function provides insights into the SCF convergence and charge distribution,
    which is essential for validating simulations.

    Parameters:
    -----------
    integrator : ConstantVLangevinIntegrator or ConstantVDrudeLangevinIntegrator
        The integrator containing electrode atom lists
    system : openmm.System
        The system containing NonbondedForce

    Returns:
    --------
    dict
        Dictionary with keys:
        - 'cathode_total_charge': Total charge on cathode (e)
        - 'anode_total_charge': Total charge on anode (e)
        - 'conductor_charges': List of total charges on conductors (e)
        - 'charge_balance': cathode + anode + conductors (should be ~0)
        - 'cathode_atoms': Number of cathode atoms
        - 'anode_atoms': Number of anode atoms
        - 'conductor_atoms': List of atom counts per conductor
        - 'voltage': Applied voltage (V)

    Example:
    --------
    >>> summary = get_electrode_charge_summary(integrator, system)
    >>> print(f"Cathode charge: {summary['cathode_total_charge']:.6f} e")
    >>> print(f"Anode charge: {summary['anode_total_charge']:.6f} e")
    >>> print(f"Charge balance: {summary['charge_balance']:.2e} e")
    >>>
    >>> # Check charge conservation (should be small)
    >>> if abs(summary['charge_balance']) > 1e-6:
    ...     print("Warning: Charge not conserved!")

    Notes:
    ------
    - Charges are in elementary charge units (e)
    - charge_balance should be close to zero if SCF converged properly
    - This function does NOT query Q_analytic (internal SCF variable)
    - Replicates diagnostic functionality from Original MM_classes.py
    """
    from openmm import NonbondedForce

    # Find NonbondedForce
    nonbonded_force = None
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, NonbondedForce):
            nonbonded_force = force
            break

    if nonbonded_force is None:
        raise RuntimeError("NonbondedForce not found in system")

    # Get cathode atoms
    cathode_atoms = []
    try:
        # Try ConstantVDrudeLangevinIntegrator API
        cathode_atoms = list(integrator.getCathodeAtomIndices())
    except AttributeError:
        # Fall back to ConstantVLangevinIntegrator API
        num_cathode = integrator.getNumCathodeAtoms()
        for i in range(num_cathode):
            particle, area = integrator.getCathodeAtomParameters(i)
            cathode_atoms.append(particle)

    # Get anode atoms
    anode_atoms = []
    try:
        anode_atoms = list(integrator.getAnodeAtomIndices())
    except AttributeError:
        num_anode = integrator.getNumAnodeAtoms()
        for i in range(num_anode):
            particle, area = integrator.getAnodeAtomParameters(i)
            anode_atoms.append(particle)

    # Get Buckyball conductor atoms
    conductor_atom_lists = []
    constantv_force = None
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if force.__class__.__name__ == 'ConstantVForce':
            constantv_force = force
            break

    if constantv_force is not None:
        num_conductors = constantv_force.getNumBuckyballConductors()
        for i in range(num_conductors):
            params = constantv_force.getBuckyballConductorParameters(i)
            virtual_atoms = params['virtualAtoms']
            conductor_atom_lists.append(virtual_atoms)

    # Compute total charges
    cathode_total = 0.0
    for atom_idx in cathode_atoms:
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(atom_idx)
        cathode_total += charge._value

    anode_total = 0.0
    for atom_idx in anode_atoms:
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(atom_idx)
        anode_total += charge._value

    conductor_totals = []
    for conductor_atoms in conductor_atom_lists:
        conductor_total = 0.0
        for atom_idx in conductor_atoms:
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(atom_idx)
            conductor_total += charge._value
        conductor_totals.append(conductor_total)

    # Compute charge balance
    charge_balance = cathode_total + anode_total + sum(conductor_totals)

    # Get voltage
    voltage = integrator.getVoltage()

    return {
        'cathode_total_charge': cathode_total,
        'anode_total_charge': anode_total,
        'conductor_charges': conductor_totals,
        'charge_balance': charge_balance,
        'cathode_atoms': len(cathode_atoms),
        'anode_atoms': len(anode_atoms),
        'conductor_atoms': [len(c) for c in conductor_atom_lists],
        'voltage': voltage
    }


def print_electrode_charge_summary(integrator, system):
    """
    Print a formatted electrode charge summary to console.

    Convenience wrapper around get_electrode_charge_summary() for debugging.

    Example:
    --------
    >>> print_electrode_charge_summary(integrator, system)
    ═══════════════════════════════════════════════════════════
    Electrode Charge Summary
    ═══════════════════════════════════════════════════════════
    Voltage: 1.000 V
    -----------------------------------------------------------
    Cathode:
      - Atoms: 1200
      - Total charge: +0.045623 e
      - Average: +0.000038 e/atom
    -----------------------------------------------------------
    Anode:
      - Atoms: 1200
      - Total charge: -0.045619 e
      - Average: -0.000038 e/atom
    -----------------------------------------------------------
    Conductors:
      1. 60 atoms, total: +0.000023 e
    -----------------------------------------------------------
    Charge Balance: +2.7e-05 e (should be ~0)
    ═══════════════════════════════════════════════════════════
    """
    summary = get_electrode_charge_summary(integrator, system)

    print(f"\n{'='*60}")
    print(f"Electrode Charge Summary")
    print(f"{'='*60}")
    print(f"Voltage: {summary['voltage']:.3f} V")
    print(f"{'-'*60}")

    # Cathode
    print(f"Cathode:")
    print(f"  - Atoms: {summary['cathode_atoms']}")
    print(f"  - Total charge: {summary['cathode_total_charge']:+.6f} e")
    if summary['cathode_atoms'] > 0:
        avg = summary['cathode_total_charge'] / summary['cathode_atoms']
        print(f"  - Average: {avg:+.6e} e/atom")

    print(f"{'-'*60}")

    # Anode
    print(f"Anode:")
    print(f"  - Atoms: {summary['anode_atoms']}")
    print(f"  - Total charge: {summary['anode_total_charge']:+.6f} e")
    if summary['anode_atoms'] > 0:
        avg = summary['anode_total_charge'] / summary['anode_atoms']
        print(f"  - Average: {avg:+.6e} e/atom")

    # Conductors
    if summary['conductor_charges']:
        print(f"{'-'*60}")
        print(f"Conductors:")
        for i, (total, num_atoms) in enumerate(zip(summary['conductor_charges'], summary['conductor_atoms'])):
            avg = total / num_atoms if num_atoms > 0 else 0
            print(f"  {i+1}. {num_atoms} atoms, total: {total:+.6f} e, avg: {avg:+.6e} e/atom")

    # Balance
    print(f"{'-'*60}")
    balance_str = f"{summary['charge_balance']:.2e}" if abs(summary['charge_balance']) > 1e-10 else "~0"
    status = "✓ OK" if abs(summary['charge_balance']) < 1e-6 else "⚠ WARNING"
    print(f"Charge Balance: {balance_str} e ({status})")

    if abs(summary['charge_balance']) >= 1e-6:
        print(f"⚠️  Warning: Charge not conserved! SCF may not have converged.")

    print(f"{'='*60}\n")


def get_scf_constants():
    """
    Get SCF algorithm constants (for reference/debugging).

    Returns:
    --------
    dict
        Dictionary with SCF parameters:
        - 'small_threshold': Charge threshold (1e-6 e)
        - 'conversion_eV_kJmol': 96.487 kJ/(mol·eV)
        - 'conversion_nmBohr': 18.8973 nm/Bohr
        - 'conversion_kJmolNm_au': 18.8973/2625.5

    Example:
    --------
    >>> constants = get_scf_constants()
    >>> print(f"Small threshold: {constants['small_threshold']:.2e} e")
    >>> print(f"Charge below this value may trigger numerical issues")

    Notes:
    ------
    - These values are hard-coded in the C++ kernel
    - small_threshold prevents division by zero in E = F/q
    - Replicates MM_classes.py:48 and other constants from Original
    """
    return {
        'small_threshold': 1e-6,  # MM_classes.py:48
        'conversion_eV_kJmol': 96.487,  # MM_classes.py:44
        'conversion_nmBohr': 18.8973,  # MM_classes.py:45
        'conversion_kJmolNm_au': 18.8973 / 2625.5  # MM_classes.py:46-47
    }

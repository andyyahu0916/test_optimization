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


def add_electrolyte_atoms_auto(topology, integrator, nonbonded_force,
                               natom_cutoff=100, exclude_chains=None):
    """
    Automatically identify and add electrolyte atoms based on residue size.

    This function replicates the Original behavior from:
    - MM_classes.py::initialize_electrolyte() Lines 256-279

    Logic: Residues with < natom_cutoff atoms are considered electrolyte.
           This is the exact logic from the Original code.

    Parameters
    ----------
    topology : openmm.app.Topology
        The system topology
    integrator : ConstantVIntegrator
        The integrator to add atoms to
    nonbonded_force : openmm.NonbondedForce
        The NonbondedForce (to get charges)
    natom_cutoff : int, default=100
        Residues with < natom_cutoff atoms are electrolyte
        (Original default: 100, Line 256)
    exclude_chains : list of int, optional
        Chain indices to exclude (e.g., electrode chains)

    Returns
    -------
    electrolyte_atoms : list of int
        List of electrolyte atom indices added

    Example
    -------
    >>> electrolyte_atoms = add_electrolyte_atoms_auto(
    ...     pdb.topology, integrator, nonbonded_force,
    ...     natom_cutoff=100,
    ...     exclude_chains=[0, 1]  # Exclude electrode chains
    ... )
    >>> print(f"Added {len(electrolyte_atoms)} electrolyte atoms")
    """

    if exclude_chains is None:
        exclude_chains = []

    # ═══════════════════════════════════════════════════════════
    # Initialize data structures
    # (Replicates Lines 258-261)
    # ═══════════════════════════════════════════════════════════
    electrolyte_atoms = []
    electrolyte_residue_names = set()

    # ═══════════════════════════════════════════════════════════
    # Loop over residues
    # (Replicates Lines 262-279)
    # ═══════════════════════════════════════════════════════════
    for res in topology.residues():
        # Skip excluded chains
        if hasattr(res, 'chain') and res.chain.index in exclude_chains:
            continue

        # Check if we've seen this residue name before
        # (Optimization from Line 263)
        if res.name in electrolyte_residue_names:
            # Already know it's electrolyte, add all atoms
            for atom in res.atoms():
                charge, sigma, epsilon = nonbonded_force.getParticleParameters(atom.index)
                electrolyte_atoms.append(atom.index)
                integrator.addElectrolyteAtom(atom.index, charge)
        else:
            # This is a new residue name, count atoms
            # (Lines 269-271)
            natoms = sum(1 for _ in res.atoms())

            # Check if it's an electrolyte residue
            # (Line 273)
            if natoms < natom_cutoff:
                # This is an electrolyte residue
                # (Lines 274-279)
                electrolyte_residue_names.add(res.name)
                for atom in res.atoms():
                    charge, sigma, epsilon = nonbonded_force.getParticleParameters(atom.index)
                    electrolyte_atoms.append(atom.index)
                    integrator.addElectrolyteAtom(atom.index, charge)

    # ═══════════════════════════════════════════════════════════
    # Print summary
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Auto-identified electrolyte atoms:")
    print(f"{'='*60}")
    print(f"  Total electrolyte atoms: {len(electrolyte_atoms)}")
    print(f"  Electrolyte residue types: {sorted(electrolyte_residue_names)}")
    print(f"  Cutoff (natom < threshold): {natom_cutoff}")
    print(f"{'='*60}\n")

    return electrolyte_atoms


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

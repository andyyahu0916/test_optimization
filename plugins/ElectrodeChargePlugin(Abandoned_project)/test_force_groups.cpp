#include <iostream>
#include "OpenMM.h"
#include "ElectrodeChargeForce.h"

using namespace OpenMM;
using namespace std;
using namespace ElectrodeChargePlugin;

int main() {
    System system;
    for (int i = 0; i < 10; i++)
        system.addParticle(1.0);

    NonbondedForce* nb = new NonbondedForce();
    nb->setNonbondedMethod(NonbondedForce::NoCutoff);
    for (int i = 0; i < 10; i++)
        nb->addParticle(0.001, 1.0, 0.0);
    system.addForce(nb);

    ElectrodeChargeForce* ecf = new ElectrodeChargeForce();
    vector<int> cathode = {0, 1, 2, 3, 4};
    vector<int> anode = {5, 6, 7, 8, 9};
    ecf->setCathode(cathode, -0.5);
    ecf->setAnode(anode, 0.5);
    ecf->setCellGap(1.5);
    ecf->setCellLength(14.0);
    ecf->setNumIterations(3);
    system.addForce(ecf);

    cout << "NonbondedForce force group: " << nb->getForceGroup() << endl;
    cout << "ElectrodeChargeForce force group: " << ecf->getForceGroup() << endl;

    return 0;
}

#include "internal/ConstantVForceImpl.h"
#include "ConstantVKernels.h"
#include "openmm/internal/ContextImpl.h"

using namespace ConstantVPlugin;
using namespace OpenMM;
using namespace std;

ConstantVForceImpl::ConstantVForceImpl(const ConstantVForce& owner) : owner(owner) {
}

ConstantVForceImpl::~ConstantVForceImpl() {
}

void ConstantVForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcConstantVKernel::Name(), context);
    kernel.getAs<CalcConstantVKernel>().initialize(context.getSystem(), owner);
}

double ConstantVForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups & (1 << owner.getForceGroup())) != 0)
        return kernel.getAs<CalcConstantVKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

vector<string> ConstantVForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcConstantVKernel::Name());
    return names;
}

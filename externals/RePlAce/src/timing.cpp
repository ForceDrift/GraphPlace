#include "timing.h"
#include <iostream>

namespace Timing {

pin::pin() {}
net::net() {}
net::net(std::string netname, double lcap, vector<int> sources, vector<int> xsinks, int _origIdx) {}

PinInfo::PinInfo() {}
PinInfo::PinInfo(PIN* curPin) {}
PinInfo::PinInfo(const PinInfo& k) {}
uint32_t PinInfo::GetData() const { return 0; }
PINNUM_TYPE PinInfo::GetPinNum() const { return 0; }
uint32_t PinInfo::GetIdx() const { return 0; }
void PinInfo::SetPinInfo(PIN* curPin) {}
void PinInfo::SetTerminal(int termIdx, PINNUM_TYPE pinIdx) {}
void PinInfo::SetModule(int moduleIdx, PINNUM_TYPE pinIdx) {}
void PinInfo::SetSteiner(PINNUM_TYPE stnIdx, int _netIdx) {}
void PinInfo::SetImpossible() {}
std::string PinInfo::GetPinName(void* ptr, vector<vector<std::string>>& pNameStor, bool isEscape) { return ""; }
std::string PinInfo::GetStnPinName(bool isEscape) { return ""; }
bool PinInfo::isTerminal() { return false; }
bool PinInfo::isModule() { return false; }
bool PinInfo::isSteiner() { return false; }
void PinInfo::Print() {}

wire::wire(PinInfo ipin, PinInfo opin, double length) {}
void wire::Print() {}

Timing::Timing(MODULE* modules, TERM* terms, NET* nets, int netCnt, PIN* pins,
       int pinCnt, 
       vector< vector< std::string > >& mPinName,
       vector< vector< std::string > >& tPinName, 
       std::string clkName, float clkPeriod) 
    : _mPinName(mPinName), _tPinName(tPinName) {}

void Timing::BuildSteiner(bool scaleApplied) {}
void Timing::WriteSpef(const std::string& spefFile) {}
void Timing::ExecuteStaFirst(std::string topCellName, std::string verilogName,
                     vector< std::string >& libName, std::string sdcName) {}
void Timing::ExecuteStaLater() {}

} // namespace Timing

long GetTimingHPWL() { return 0; }

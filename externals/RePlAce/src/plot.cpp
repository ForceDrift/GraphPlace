#include "plot.h"
#include <iostream>

using namespace std;

// Dummy implementations to remove CImg/JPEG/X11 dependencies
void SaveCellPlotAsJPEG(string title, bool isMacroOnly, string fileName) {
    cout << "[PLOTTING DISABLED] " << title << " -> " << fileName << endl;
}

void SaveDensityPlotAsJPEG(string title, string fileName) {
    // dummy
}

void SaveFieldPlotAsJPEG(string title, string fileName) {
    // dummy
}

void SaveArrowPlotAsJPEG(string title, string fileName) {
    // dummy
}

void save_jpeg(string fileName) {
    // dummy
}

void SaveBinPlotAsJPEG(string title, string fileName) {
    // dummy
}

string intoFourDigit(int idx) {
    char buf[10];
    sprintf(buf, "%04d", idx);
    return string(buf);
}

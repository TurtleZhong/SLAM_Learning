#include "param_reader.h"
#include <iostream>

int main(int argc, char *argv[])
{

    ParameterReader pd;
    string use_tum = pd.getData("use_tum");
    cout << "use_tum = " << use_tum << endl;

    double gridsize = atof( pd.getData( "voxel_grid" ).c_str() );
    cout << "gridsize = " << gridsize << endl;

    return 0;
}

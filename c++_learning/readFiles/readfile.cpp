#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

int main()
{
//    double data[5];
//    for(int i = 0; i < 5; i++)
//    {
//        cin >> data[i];
//    }

//    for(int j =0; j < 5; j++)
//    {
//        cout << data[j] << endl;
//    }


    ifstream fin;
    fin.open("../pose.txt");
    if(!fin)
    {
        cout << "Error to open the file!" << endl;
    }

    double data[7];
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 7; j++)
        {
            fin >> data[j];
        }

        cout << "***********" << endl;
        for (int j = 0; j < 7; j++)
        {
            cout << data[j] << endl;
        }

    }


    return 0;
}

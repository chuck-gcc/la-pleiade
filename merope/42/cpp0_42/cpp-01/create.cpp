#include "phone.hpp"

void get_menu(int param)
{
    if(param > 3 || param < 0)
        return ;
    if(param == 0)
        cout << "First Name: ";
    if(param == 1)
        cout << "Nick Name: ";
    if(param == 2)
        cout << "Phone: ";
    if(param == 3)
        cout << "Secret: ";
}

Contact *create_contact(int id)
{
    string str_arr[4];
    string input;
    string str;
    int i;
    size_t len;
    Contact *new_contact;

    i = 0;
    while (i < 4)
    {   
        get_menu(i);
        cin >> input;
        len = input.length();
        str = input.substr(0,len);
        str_arr[i] = str;
        i++;
    }
    new_contact = new Contact(id, str_arr[0],str_arr[1],str_arr[2],str_arr[3]);
    cout << "\nNouveau contact creé ID: " << id + 1 << endl;
    return(new_contact);
}

#include "phone.hpp"

int main(void)
{
    
    PhoneBook phone(REPERTOIRE_SIZE);
    string input;
    int exit;
    int idx;

    exit = 0;
    while (!exit)
    {
        cin >> input;
        if(!std::strcmp(input.c_str(), "ADD"))
        {
            cout << endl;
            idx = REPERTOIRE_SIZE - phone.repertoire_size;
            Contact *c  =  create_contact(idx);
            phone.contact[REPERTOIRE_SIZE - phone.repertoire_size] = c;
            if(phone.repertoire_size > 0)
                phone.repertoire_size--;
            
        }
        if(!std::strcmp(input.c_str(), "SEARCH")|| !std::strcmp(input.c_str(), "S") )
            search_contact(&phone, 0);
        if(!std::strcmp(input.c_str(), "EXIT"))
            return(0);
        else
            continue;

    }
    return(0);
}
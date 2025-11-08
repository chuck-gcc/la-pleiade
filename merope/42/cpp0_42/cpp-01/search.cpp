#include "phone.hpp"

static string tmp_contact(string str)
{
    string tmp;
    if(str.length() <= 11)
        return(str);
    tmp = str.substr(0,10);
    tmp[9] = '.';
    return (tmp);
}


static Contact *format_tmp_contact(Contact *contact)
{

    if (!contact) return (NULL);

   Contact *tmp = new Contact(contact->id + 1, tmp_contact((contact->first_name)) ,
            tmp_contact((contact->nick_name)),
            tmp_contact((contact->phone_number)),
            tmp_contact((contact->secret)));
    
    
    return(tmp);
}

static void display_contact(Contact *contact)
{
    cout << "we are here\n";

    if(contact)
    {
        cout << "ID: " << contact->id << endl;
        cout << "FIRST NAME: " << contact->first_name << endl;
        cout << "NICK NAME: " << contact->nick_name << endl;
        cout << "PHONE NUMBER: " << contact->phone_number << endl;
        cout << "SECRET: " << contact->secret << endl;
        cout << endl;
    }
}
static void display_contact_list( Contact *contact, int idx)
{

    Contact *tmp = format_tmp_contact(contact);

    cout << setw(10) << tmp->id;
    cout << " | " ;
    cout << setw(10) << tmp->first_name;
    cout << " | " ;
    cout << setw(10) << tmp->nick_name;
    cout << " | " ;
    cout << setw(10) << tmp->phone_number;
    cout << " | " ;
    cout << setw(10) << tmp->secret << endl;

    delete tmp;
}

void search_contact(PhoneBook *phone, int idx)
{
    int i;
    string input;

    i = -1;
    cout << "\nNombre de contact " << REPERTOIRE_SIZE - phone->repertoire_size << endl;
    cout << endl;
    if(REPERTOIRE_SIZE - phone->repertoire_size == 0)
    {
        cout << "Not contact in repertory. Add contact" << endl;
        return ;
    }
    while (i < REPERTOIRE_SIZE - phone->repertoire_size)
    {
        if(i == -1)
        {
            cout << "------------------------------------" << endl;
            i++;
        }
        display_contact_list(phone->contact[i], i);
        i++;
    }
    cout << endl << "Choose contact index: ";
    i = 0;
    cin >> input;
    cout << endl;
    if(input.length() > 1 || !isdigit(input[0]) || idx == 2)
    {
        if(idx == 2)
            cout << "Err: 3 tentatives" << endl;
        return ;
    }
    i = std::atoi(input.c_str());
    if( i <= 0 || i > REPERTOIRE_SIZE - phone->repertoire_size)
    {
        cout << "incorrect value only "  <<  REPERTOIRE_SIZE - phone->repertoire_size << " contact in repertoire"  << endl;
        cout << "Try " << idx + 1 << " of 3";
        search_contact(phone, idx + 1);
        return ;
    }
    cout << phone->contact[i]->id;
    cout << "we arehere" << endl;
    // error from here
    display_contact(phone->contact[i]);
}
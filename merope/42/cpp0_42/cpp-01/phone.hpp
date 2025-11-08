#ifndef PHONE_H
#define PHONE_H

#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <iomanip>
#include <assert.h>
#include <unistd.h>

#define REPERTOIRE_SIZE 8
#define MAX_LEN 31
#define MAX_SECRET 256
#define PHONE_NUMBER_SIZE 15


using namespace std;

class Contact{
    public:
        string first_name;
        string nick_name;
        string secret;
        string phone_number;
        int id;
        Contact(int id, string first_name, string nick_name, string secret,string phone_number)
        {
            this->first_name = first_name;
            this->nick_name = nick_name;
            this->phone_number = phone_number;
            this->secret = secret;
            this->id = id;

        }
};

class PhoneBook{
    public:
        Contact *contact[REPERTOIRE_SIZE];
        int repertoire_size;
        PhoneBook(int size)
        {
            this->repertoire_size = size;
        }; 
};

void    search_contact(PhoneBook *phone, int idx);
Contact *create_contact(int id);

#endif
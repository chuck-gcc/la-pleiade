input::= line
line::= <cmd> {<pipe> <cmd>}
cmd ::= <exec> { <arg> } { <redirection> }
arg::= (<word> | <var>)
exec::= <word>
var::= "$<word>"
pipe::= "|"
redirection::= ("<" | ">" | ">>" | "<<") <filename>
space::= " "
filename::= <word>

input::= <pair> {newline <pair>}
pair::= <float> ";" <float>
float::= <digit>+ "."<digit>
digit:== ('0' | '1' | '2' | '3' | '4'| '5'| '6'| '7'| '8' | '9')
newline:== '\n'


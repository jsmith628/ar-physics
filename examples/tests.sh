#!/bin/sh

#cd to the examples directory (necessary for when run outside of the workspace dir)
cd `dirname $0`/..

#get all of the example configs in the examples folder
tests=`find ./examples -type f -name '*.toml'`

#loop forever
while [ 1 ]
do

  #
  #construct the menu
  #

  #the header
  echo
  echo 'Available tests:'

  #make an indexed line for ever example file
  i=0
  for config in $tests
  do
    #increment the index
    i=`expr $i + 1`

    #get the simulation name
    sim_name=`cat $config | pcre2grep --match-limit=1 'name' | pcre2grep -o '(?<=")[^"]*(?=")' `

    #output an indexed list entry
    echo "$i. $sim_name"
  done

  #
  #get the user test selection
  #

  echo -n 'Select test to run: '
  while [ 1 ]
  do

    index=`head -n 1 -` #gets the next line of stdin

    #test if the input is a number
    if echo $index | egrep -q '^[0-9]+$' && [ $index -le $i ]
    then

      #get the selected filename
      config=`echo $tests | cut -d " " -f $index`

      #get the preferred tickrate
      if cat $config | egrep -q 'preferred_tickrate'
      then
        tickrate=`cat $config | pcre2grep --match-limit=1 'preferred_tickrate' | pcre2grep -o '[+\-\d.]+'`
      else
        tickrate=-1
      fi

      #run the simulation
      cargo run --release --example test -- $config -f -1 -t $tickrate > /dev/null 2>&1
      break

    else

      #else, print an error message
      echo Invalid index!
      echo -n "Please enter a number from 1 to $i: "

    fi

  done

done

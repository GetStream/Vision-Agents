## Simulate & Test

When you create a simulation you can select a judge. 
This is just an LLM model, so use the same LLM routing logic capabilities we already have on go

When you create a simulation you can setup these fields

* Name
* Type: Text (default) or Audio (which means we will generate audio, and run through the full pipeline)
* What to ask: In 3 steps do the following: place an order for pasta bolognese, after the order is handled change your mind and change it into pepperoni pizza. tell them to deliver at 8pm
* Variations: None, Expand 10x (have AI try 10 different ways)
* Evaluation: was an order placed with delivery time 8pm for a pepperoni pizza?
* Agent (the agent to run this against)

So important capabilities here:
- The what to ask can handle multiple turns

## Dashboard

Add a simulations page in the header which shows the simulations. Have a tab on that page to see simulation logs
And inside of a simulation a button to run the simulation, and to see past results

## Simulation log

Store the results of how the simulations went. 


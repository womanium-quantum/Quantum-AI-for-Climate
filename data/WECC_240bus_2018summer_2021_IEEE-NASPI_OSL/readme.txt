The reduced WECC 240-bus system model can be found in the same folder. This model was developed by National Renewable Energy Laboratory [1] and modified by IEEE-NASPI Oscillation Source Location (OSL) Committee. See more details about system information and the modifications below. 

The .raw format power flow data and .dyr dynamic data of this system were prepared in PSSE v34 format. The system has:
 - 243 buses
 - 146 generating units at 56 power plants (including 109 synchronous machines and 37 renewable generators )
 - 329 transmission lines
 - 122 transformers
 - 7 switched shunts
 - 139 loads

The OSL committee made some changes on the system in [1] to generate the test system and test cases suitable for this contest. The change included in the download here is: 10 IEEEST PSS models added. Changes NOT included in the download are: (i) the contest cases used a different power flow dispatch and topology; (ii) Some control settings may be modified to produce the desired oscillatory phenomena; and (iii) HVDC dynamic model added for two DC lines, i.e. PDCI and IPP (see details in the case description part at http://web.eecs.utk.edu/~kaisun/Oscillation/Contest.html).

[1] H. Yuan, R. Sen biswas, J. Tan, Y. Zhang, "Developing a Reduced 240-Bus WECC Dynamic Model for Frequency Response Study of High Renewable Integration," 2020 T&D

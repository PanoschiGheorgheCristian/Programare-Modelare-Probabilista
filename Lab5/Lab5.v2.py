from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Aici am modificat valorile din Lab5.py pentru un rezultat diferit (din punct de vedere numeric)

model = BayesianNetwork([("Cartej1", "Cartej2"), 
                         ("Cartej1", "Deciziej1"), 
                         ("Deciziej1", "Deciziej2"), 
                         ("Cartej2", "Deciziej2"), 
                         ("Cartej1", "Deciziej1_prim"), 
                         ("Deciziej2", "Deciziej1_prim"),
                         ("Cartej1", "Rezultat"),
                         ("Cartej2", "Rezultat"),
                         ("Deciziej1", "Rezultat"),
                         ("Deciziej2", "Rezultat"),
                         ("Deciziej1_prim", "Rezultat")])

# A = 0, Kf = 1, Ki = 2, Qf = 3, Qi = 4

cpd_cartej1 = TabularCPD("Cartej1", 5, [[0.2], [0.2], [0.2], [0.2], [0.2]])
cpd_cartej2 = TabularCPD("Cartej2", 5, [ [0, 0.25, 0.25, 0.25, 0.25], 
                                        [0.25, 0, 0.25, 0.25, 0.25], 
                                        [0.25, 0.25, 0, 0.25, 0.25],
                                        [0.25, 0.25, 0.25, 0, 0.25], 
                                        [0.25, 0.25, 0.25, 0.25, 0] ],
                         evidence=["Cartej1"],
                         evidence_card=[5]
                        )

# Decizie = 0 <=> jucatorul a asteptat    

cpd_deciziej1 = TabularCPD("Deciziej1", 2, [ [0, 0.15, 0.5, 0.85, 1], 
                                            [1, 0.85, 0.5, 0.15, 0] ], 
                           evidence=["Cartej1"],
                           evidence_card=[5]
                           )
cpd_deciziej2 = TabularCPD("Deciziej2", 2, [ [0, 0, 0, 0.20, 0.20, 0.80, 0.80, 1, 1, 1],
                                            [1, 1, 1, 0.80, 0.80, 0.20, 0.20, 0, 0, 0] 
                                            ],
                           evidence=["Cartej2", "Deciziej1"],
                           evidence_card=[5,2]
                           )
cpd_deciziej1_prim = TabularCPD("Deciziej1_prim", 2, [ [0, 0, 0, 0.20, 0, 0.80, 0.5, 1, 1, 1],
                                                      [1, 1, 1, 0.80, 1, 0.20, 0.5, 0, 0, 0] ],
                                evidence=["Cartej1", "Deciziej2"],
                                evidence_card=[5,2]
                                )

# Rezultat = 0 <=> Nici unul nu a pariat
# Rezultat = 1 <=>   J1 a pariat si J2 a stat
# Rezultat = 2 <=> J2 a pariat si J1 a stat
# Rezultat = 3 <=> Ambii au pariat si J1 a castigat
# Rezultat = 4 <=> Ambii au pariat si J2 a castigat

cpd_rezultat = TabularCPD("Rezultat", 5, [ [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            
                                            
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                          
                                           [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                            
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 
                                            
                                            
                                            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                            
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                           
                                           [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            
                                            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 
                                            
                                            
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                           
                                           [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,0,1, 
                                            
                                            
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            
                                            1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,0,1, 
                                            1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,0,1],
                                           
                                           [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,1,0, 
                                            
                                            
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                            
                                            0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,1,0, 
                                            0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,1,0]
                                        ],
                                evidence=["Deciziej1", "Deciziej2", "Deciziej1_prim", "Cartej1", "Cartej2"],
                                evidence_card=[2,2,2,5,5]
                                )

model.add_cpds(cpd_cartej1, cpd_cartej2, cpd_deciziej1, cpd_deciziej2, cpd_deciziej1_prim, cpd_rezultat)
model.get_cpds()

print(model.check_model())

#Cazuri favorabile J1: Rezultat = 1, Rezultat = 3
#Cazuri favorabile J2: Rezultat = 2, Rezultat = 4
#Caz neutru: Rezultat=0

infer = VariableElimination(model)
posterior_p1 = infer.query(["Rezultat"], evidence={"Cartej1" : 1})
print(posterior_p1)
# Jucatorul 1 ar trebui sa joace pentru ca are sanse mari sa castige macar 1$

posterior_p2 = infer.query(["Rezultat"], evidence={"Cartej2" : 2, "Deciziej1" : 1})
print(posterior_p2)
# Jucatorul 2 ar trebui sa stea pentru ca are sanse mari sa piarda mai mult de 1$
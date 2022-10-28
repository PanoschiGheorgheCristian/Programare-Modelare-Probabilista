from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

model = BayesianNetwork([("Cartej1", "Cartej2"), 
                         ("Cartej1", "Deciziej1"), 
                         ("Deciziej1", "Deciziej2"), 
                         ("Cartej2", "Deciziej2"), 
                         ("Cartej1", "Deciziej1_prim"), 
                         ("Deciziej2", "Deciziej1_prim")])

cpd_cartej1 = TabularCPD("Cartej1", 5, [[0.2], [0.2], [0.2], [0.2], [0.2]])
cpd_cartej2 = TabularCPD("Cartej2", 5, [ [0, 0.25, 0.25, 0.25, 0.25], 
                                        [0.25, 0, 0.25, 0.25, 0.25], 
                                        [0.25, 0.25, 0, 0.25, 0.25],
                                        [0.25, 0.25, 0.25, 0, 0.25], 
                                        [0.25, 0.25, 0.25, 0.25, 0] ],
                         evidence=["Cartej1"],
                         evidence_card=[5]
                        )
cpd_deciziej1 = TabularCPD("Deciziej1", 2, [ [0, 0.25, 0.5, 0.75, 1], 
                                            [1, 0.75, 0.5, 0.25, 0] ], 
                           evidence=["Cartej1"],
                           evidence_card=[5]
                           )
cpd_deciziej2 = TabularCPD("Deciziej2", 2, [ [0, 0, 0, 0.33, 0.33, 0.66, 0.66, 1, 1, 1],
                                            [1, 1, 1, 0.66, 0.66, 0.33, 0.33, 0, 0, 0] 
                                            ],
                           evidence=["Cartej2", "Deciziej1"],
                           evidence_card=[5,2]
                           )
cpd_deciziej1_prim = TabularCPD("Deciziej1_prim", 2, [ [0, 0, 0, 0.33, 0, 0.66, 0.5, 1, 1, 1],
                                                      [1, 1, 1, 0.66, 1, 0.33, 0.5, 0, 0, 0] ],
                                evidence=["Cartej1", "Deciziej2"],
                                evidence_card=[5,2]
                                )

model.add_cpds(cpd_cartej1, cpd_cartej2, cpd_deciziej1, cpd_deciziej2, cpd_deciziej1_prim)
model.get_cpds()

print(model.check_model())
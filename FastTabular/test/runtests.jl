using FastAI, FastTabular, ReTest

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

FastTabular.runtests([ReTest.fail, ReTest.not(ReTest.pass)])

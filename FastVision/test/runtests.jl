using FastAI, FastVision, ReTest

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

FastVision.runtests([ReTest.fail, ReTest.not(ReTest.pass)])

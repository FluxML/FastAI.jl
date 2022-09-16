using FastAI, FastTimeSeries, ReTest

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

FastTimeSeries.runtests([ReTest.fail, ReTest.not(ReTest.pass)])
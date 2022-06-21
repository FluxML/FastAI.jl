using FastAI, FastVision, ReTest

FastVision.runtests([ReTest.fail, ReTest.not(ReTest.pass)])

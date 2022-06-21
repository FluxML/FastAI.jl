using FastAI, FastTabular, ReTest


FastTabular.runtests([ReTest.fail, ReTest.not(ReTest.pass)])

(define (problem rearrange3b4l)
  (:domain rearrange-blocks-with-costume)
  
  (:objects 
    block0 block1 block2 can - block
    pos1 pos2 pos3 pos4 - location
  )

  (:init
    (at block0 pos1)
    (at block1 pos2)
    (at block2 pos3)
    (at can pos4)
    (handempty)
    (occupied pos1)
    (occupied pos2)
    (occupied pos3)
    (occupied pos4)
  )

  (:goal
    (and
      (at block0 pos2)
      (at block1 pos1)
      (at block2 pos3)
      (at can pos4)
    )
  )
)
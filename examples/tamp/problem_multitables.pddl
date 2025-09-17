(define (problem grocery-sorting)

    (:domain rearrange-s3e)
    (:objects
        bread can - block
        green-table blue-table - location
    )

    (:init

        (handempty)
        (at bread green-table)
        (at can green-table)
    )

    (:goal
        (and
            (at bread blue-table)
        )
    )
)
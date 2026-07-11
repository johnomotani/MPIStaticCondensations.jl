struct BlockedSchurComplementSolver{TA,TB,TC,TS,TSF,TAiu,Ttimer}
    A_factorization::TA
    B::TB
    C::TC
    schur_complement::TS
    schur_complement_factorization::TSF
    Ainv_dot_u::TAiu
    timer::Ttimer
end

function lu!(sc::BlockedSchurComplementSolver, full_A)
    timer = sc.timer
    A_factorization = sc.A_factorization
    B = sc.B
    C = sc.C
    schur_complement = sc.schur_complement
    schur_complement_factorization = sc.schur_complement_factorization
    synchronize_shared = sc.synchronize_shared

    @sc_timeit timer "lu! BlockedSchurComplementSolver" begin
        @sc_timeit timer "lu(A)" begin
            lu!(A_factorization, full_A)
        end
        @sc_timeit timer "Ainv_dot_B" begin
            synchronize_shared()

            copy_B_submatrix!(B, full_A)

            ldiv_Bmatrix!(A_factorization, B)
        end
        @sc_timeit timer "C" begin
            copy_C_submatrix!(C, full_A)
        end
        @sc_timeit timer "schur_complement" begin
            synchronize_shared()

            mul_C_Ainv_dot_B!(schur_complement, C, B)
            synchronize_shared()
            add_D_to_schur_complement!(schur_complement, full_A)
            synchronize_shared()

            lu!(schur_complement_factorization, schur_complement.matrix)
        end
    end

    return nothing
end

function ldiv!(X::AbstractVector, y::AbstractVector, sc::BlockedSchurComplementSolver,
               U::AbstractVector, v::AbstractVector)
    @inbounds begin
        timer = sc.timer
        @sc_timeit timer "ldiv!" begin
            A_factorization = sc.A_factorization
            B = sc.B
            schur_complement_factorization = sc.schur_complement_factorization
            Ainv_dot_u = sc.Ainv_dot_u
            top_vec_buffer = sc.top_vec_buffer
            bottom_vec_buffer = sc.bottom_vec_buffer
            global_bottom_vector_range_partial = sc.global_bottom_vector_range_partial
            global_bottom_vector_entries_no_overlap_partial = sc.global_bottom_vector_entries_no_overlap_partial
            local_bottom_vector_range_partial = sc.local_bottom_vector_range_partial
            local_bottom_vector_entries_no_overlap_partial = sc.local_bottom_vector_entries_no_overlap_partial
            schur_complement_local_range_partial = sc.schur_complement_local_range_partial
            shared_rank = sc.shared_rank
            distributed_nproc = sc.distributed_nproc
            synchronize_shared = sc.synchronize_shared

            @sc_timeit timer "Ainv.u" begin
                ldiv!(Ainv_dot_u, A_factorization, U)
            end

            @sc_timeit timer "v-C.Ainv.u" begin
                # Initialise to zero, because when C does not include all rows, the matrix
                # multiplication below would not initialise all elements.
                bottom_vec_buffer[schur_complement_local_range_partial] .= 0.0
                synchronize_shared()
                mul_C_dot_Ainv_dot_u!(bottom_vec_buffer, sc.C, Ainv_dot_u)
                synchronize_shared()

                # Only have the local entries of v, so add those to the local entries in
                # bottom_vec_buffer before reducing.
                # Need to avoid double counting of any overlapping entries in `v`.
                for (i1, i2) ∈ zip(global_bottom_vector_entries_no_overlap_partial, local_bottom_vector_entries_no_overlap_partial)
                    bottom_vec_buffer[i1] += v[i2]
                end
                synchronize_shared()
            end

            @sc_timeit timer "global_y" begin
                # `global_y` is solved in serial on the global rank-0 process, and then
                # communicated back to all other processes.
                global_y = sc.global_y
                if sc.shared_rank == 0 && distributed_nproc > 1
                    MPI.Reduce!(bottom_vec_buffer, +, distributed_comm; root=0)
                end

                if distributed_nproc > 1
                    synchronize_shared()
                end
                ldiv!(global_y, schur_complement_factorization, bottom_vec_buffer)
                if distributed_nproc > 1
                    synchronize_shared()
                end

                if sc.shared_rank == 0 && distributed_nproc > 1
                    MPI.Bcast!(global_y, distributed_comm; root=0)
                end
                synchronize_shared()
            end

            @sc_timeit timer "Ainv.u-Ainv.B.y" begin
                Ainv_dot_u_minus_Ainv_dot_B_dot_y!(top_vec_buffer, Ainv_dot_u, B, global_y)

                for (i1, i2) ∈ zip(local_bottom_vector_range_partial, global_bottom_vector_range_partial)
                    y[i1] = global_y[i2]
                end
                synchronize_shared()
            end
        end

        return nothing
    end
end

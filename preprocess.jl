using Distributions, SparseArrays, LinearAlgebra,DataStructures, MathOptInterface

function preprocess!(m)
    # Copy over model to P
    P = factorable!(m)	 
    quadObj = nothing
    colVal = []
    
    # provide initial value if not defined
    # Copy over initial variable values
    for var in JuMP.all_variables(P)
        if isnothing(var)
           JuMP.set_start_value(var, 0)
        end
        push!(colVal,JuMP.start_value(var))
    end
    
    # Set the objective value as a variable to minimize and
    # move the objective to a constraint. Eval_g gets the
    # initial value of the objective from variable values
    quadObj = JuMP.objective_function(P)
    objSense = JuMP.objective_sense(P)
    print(objSense)
    @variable(P, objective_value, start=eval_g(quadObj, colVal))
    if objSense == :Min
        @constraint(P, objective_value==quadObj)
        @objective(P, Min, objective_value)
    else
        @constraint(P, objective_value==quadObj)
        @objective(P, Max, objective_value)
    end
    
    # provide bounds on variables if not defined
    for var in JuMP.all_variables(P)
       if !JuMP.has_lower_bound(var)
            JuMP.set_lower_bound(var,default_lower_bound_value)
       end
       if !JuMP.has_upper_bound(var)
            JuMP.set_upper_bound(var,default_upper_bound_value)
       end
    end
    
    # Gather all the quadratic constraints
    quadConstr = []
    sets = [MathOptInterface.EqualTo{Float64},MathOptInterface.LessThan{Float64},MathOptInterface.GreaterThan{Float64}]
    for set in sets
        append!(quadConstr,JuMP.all_constraints(P,GenericQuadExpr{Float64,VariableRef},set))
    end
    
    # modify the constraints, so that Q is upper triangular
    for quad in quadConstr
        quad = JuMP.constraint_object(quad)
    	squeeze!(quad.func)
        terms = quad.func.terms
        qvars1 = [pair.a for (pair,coeff) in terms]
        qvars2 = [pair.b for (pair,coeff) in terms]
        qcoeffs = [coeff for (pair,coeff) in terms]
        qVarsId_local = []
        
        # Get the variable indices of the quadratic variables
        for j in 1:length(qvars1)
            push!(qVarsId_local, [qvars1[j].index.value, qvars2[j].index.value])
        end
        
        j = 1
        # If another constraint has the same quadratic constraint, need to
        # combine with other constraint
        while j <= length(qvars1)
            var1 = qvars1[j]
            var2 = qvars2[j]
            
            if var1.index.value > var2.index.value
                qvars1[j] = var2
                qvars2[j] = var1
            end
            
            # Get the index of these quadratic variables
            index = findall( x->(x == [var1, var2] || x == [var2, var1]), qVarsId_local)
            
            # If this quadratic variable set shows up again in the local expression,
            # combine into a single constraint
            if length(index) > 1
               qcoeffs[j] = sum(qcoeffs[index])
               deleteat!(index, findall(x->x==j,index))
               deleteat!(qVarsId_local, index)
               deleteat!(qcoeffs, index)
               deleteat!(qvars1, index)
               deleteat!(qvars2, index)
            end
            j += 1
        end
        pairs = [UnorderedPair(qvar1,qvar2) for (qvar1,qvar2) in zip(qvars1,qvars2)]
        terms_new = OrderedDict(zip(pairs,qcoeffs))
        quad.func.terms = terms_new
    end


    # create a simple relaxed problem
    m = Model() #solver=GurobiSolver(Threads=1, LogToConsole=0))#, LogFile=string(runName,".txt")))
    m.ext[:v_map] = Dict{VariableRef,VariableRef}()
    m.ext[:fstg_n] = Dict{String,VariableRef}()
    m.ext[:idx_map] = Dict{Int64,MOI.VariableIndex}()
    
    # Variables
    for var in JuMP.all_variables(P)
        new_x = JuMP.@variable(m)            #create an anonymous variable
        push!(m.ext[:v_map], var => new_x)
        var_name = JuMP.name(var)
        push!(m.ext[:fstg_n],var_name=>new_x)
        push!(m.ext[:idx_map],new_x.index.value=>new_x.index)
        new_name = var_name
        JuMP.set_name(new_x,new_name)
        if JuMP.start_value(var) != nothing
            JuMP.set_start_value(new_x,JuMP.start_value(var))
        end
     end

    # Constraints
    constraint_types = JuMP.list_of_constraint_types(P)
    quad_types = Dict{Any,Any}()
    quad_length = length(collect(keys(quad_types)))
    
    for (func,set) in constraint_types
        if func==GenericQuadExpr{Float64,VariableRef}
            push!(quad_types,func=>set)
        else
            constraint_refs = JuMP.all_constraints(P, func, set)
            for constraint_ref in constraint_refs
                constraint = JuMP.constraint_object(constraint_ref)
                new_constraint = PlasmoOld._copy_constraint(constraint,m)
                new_ref= JuMP.add_constraint(m,new_constraint)
            end
        end
    end
    
    #OBJECTIVE FUNCTION (store expression on combinedd_nodes)
    if !(PlasmoOld._has_nonlinear_obj(P))
        #AFFINE OR QUADTRATIC OBJECTIVE
        new_objective = PlasmoOld._copy_objective(P,m.ext[:v_map])
        objSense = objective_sense(P)
        set_objective_function(m,new_objective)
        set_objective_sense(m,objSense)
    else
        #NONLINEAR OBJECTIVE
        print("sorry, currently not support nonlinear objective function, please formulate it as a constraint!")
    end
    
    # constraints
    branchVarsId = []
    qbVarsId = Dict()
    bilinearVars = []
    i=0
    for (func,set) in quad_types
        constraint_refs = JuMP.all_constraints(P, func, set)
        for constraint in constraint_refs
            con = JuMP.constraint_object(constraint)
            new_constraint = PlasmoOld._copy_constraint(con,m)
            new_ref= JuMP.add_constraint(m,new_constraint)
            
            qvars1 = [pair.a for (pair,coeff) in new_constraint.func.terms]
            qvars2 = [pair.b for (pair,coeff) in new_constraint.func.terms]
            
            for j in 1:length(qvars1)
                if !haskey(qbVarsId, (qvars1[j].index.value, qvars2[j].index.value))
                   push!(branchVarsId, qvars1[j].index.value)
                   push!(branchVarsId, qvars2[j].index.value)
                   bilinear = @variable(m)
                   varname = " bilinear_con$(i)_"*string(qvars1[j])*"_"*string(qvars2[j])*"_$(j)"
                   JuMP.set_name(bilinear, varname)
                   m.ext[:idx_map][bilinear.index.value] = bilinear.index
                   push!(bilinearVars, bilinear)
                   qbVarsId[qvars1[j].index.value, qvars2[j].index.value] = bilinear.index.value
                   qbVarsId[qvars2[j].index.value, qvars1[j].index.value] = bilinear.index.value
                end
            end
        i+=1
        end
    end
    
    branchVarsId = sort(union(branchVarsId))
    
    # Need to add handling for nonlinear constraints.
    # Need functionality to exist in the extensive function
    # maker before implementing it here.
    
    #num_cons = MathProgBase.numconstr(P)
    expVariable_list = []
    logVariable_list = []
    powerVariable_list = []
    monomialVariable_list = []
    """
    d = JuMP.NLPEvaluator(P)
    MathProgBase.initialize(d,[:ExprGraph])
    for i = (1+length(P.linconstr)+length(P.quadconstr)):num_cons
    	expr = MathProgBase.constr_expr(d,i)
	if isexponentialcon(expr)
	   ev = parseexponentialcon(expr)
	   push!(expVariable_list, ev)
        elseif islogcon(expr)
           ev = parselogcon(expr)
           push!(logVariable_list, ev)
        elseif ispowercon(expr)
           ev = parsepowercon(expr)
           push!(powerVariable_list, ev)
        elseif ismonomialcon(expr)
           ev = parsemonomialcon(expr)
	   push!(expVariable_list, ev)
           #push!(monomialVariable_list, ev)
        end
    end
    """
    
    Pcopy = PlasmoOld.copyNet(P)
    multiVariable_list = []
    multiVariable_convex = []
    multiVariable_aBB = []
    
    # Constraints
   constraint_types = JuMP.list_of_constraint_types(P)
   quad_types = Dict{Any,Any}()
   
    for (func,set) in constraint_types
        if func==GenericQuadExpr{Float64,VariableRef}
            push!(quad_types,func=>set)
        end
    end
    
    for (func,set) in quad_types
        quads = JuMP.all_constraints(Pcopy,func,set)
        for i = 1:length(quads)
            con = JuMP.constraint_object(quads[i])
            terms = con.func.terms
            qvars1 = [pair.a for (pair,coeff) in terms]
            qvars2 = [pair.b for (pair,coeff) in terms]
            qcoeffs = [coeff for (pair,coeff) in terms]
            aff = con.func.aff
            
            mvs = MultiVariable[]
            remain=collect(1:length(qvars1))
            while length(remain) != 0
                mv = MultiVariable()
                mvTerms = mv.terms
                push!(mvs, mv)
                seed = remain[1]
                var1 = qvars1[seed]
                var2 = qvars2[seed]
                coeff = qcoeffs[seed]
                arr1 = [JuMP.UnorderedPair(var1,var2),]
                arr2 = [coeff,]
                merge!(mvTerms.terms, OrderedDict(zip(arr1,arr2)))
                push!(mv.qVarsId, var1.index.value)
                push!(mv.qVarsId, var2.index.value)
                mv.qVarsId = sort(union(mv.qVarsId))
                deleteat!(remain, 1)
            while length(remain) != 0
                added = []
                for k in 1:length(remain)
                    j = remain[k]
                    var1 = qvars1[j]
                    var2 = qvars2[j]
                    coeff = qcoeffs[j]
                    index = findall((in)(var1.index.value),mv.qVarsId)
                    if index == []
                        index = findall((in)(var2.index.value),mv.qVarsId)
                    end
                    if length(index) != 0
                        merge!(mvTerms.terms, OrderedDict(zip(JuMP.UnorderedPair(var1,var2),coeff)))
                        push!(mv.qVarsId, var1.index.value)
                        push!(mv.qVarsId, var2.index.value)
                        mv.qVarsId = sort(union(mv.qVarsId))
                        push!(added, k)
                    end
                end
                deleteat!(remain, added)
                if length(added) == 0
                    break
                end
            end
            end

            nonlinearIndexs = []
            affvars = [var for (var,coeff) in aff.terms]
            affcoeffs = [coeff for (var,coeff) in aff.terms]
            
            for j in 1:length(affvars)
                var = affvars[j]
                coeff = affcoeffs[j]
                index = findall( x->( length(findall((in)(var.index.value),x.qVarsId))!= 0), mvs)
                if length(index) != 0
                    mv = mvs[index[1]]
                    term1 = [var,]
                    term2 = [coeff,]
                    merge!(mv.terms.aff.terms,OrderedDict(zip(term1,term2)))
                    m.ext[:idx_map][var.index.value] = var.index
                    push!(nonlinearIndexs, j)
                end
            end
            
            deleteat!(affvars, nonlinearIndexs)
            deleteat!(affcoeffs, nonlinearIndexs)
            copyTerms = OrderedDict(zip(affvars,affcoeffs))
            affcopy = GenericAffExpr(aff.constant,copyTerms)
            
            for j in 1:length(mvs)
                mv = mvs[j]
                mvTerms = mv.terms
                C = length(mv.qVarsId)
                Q = zeros(Float64, C, C)

                qvars1 = [pair.a for (pair,coeff) in mv.terms.terms]
                qvars2 = [pair.b for (pair,coeff) in mv.terms.terms]
                qcoeffs = [coeff for (pair,coeff) in mv.terms.terms]
        
                for k in 1:length(qvars1)
                    var1 = qvars1[k]
                    var2 = qvars2[k]
                    coeff = qcoeffs[k]
                    I = findall((in)(var1.index.value),mv.qVarsId)[1]
                    J = findall((in)(var2.index.value),mv.qVarsId)[1]
                    Q[I, J] = coeff
                    if haskey(qbVarsId, (var1.index.value, var2.index.value))
                        bid = qbVarsId[var1.index.value, var2.index.value]
                    end
                    push!(mv.bilinearVarsId, bid)
                end
                mv.Q = copy(Q)
                Q = (Q+Q')/2
                val, vec = (eigen(Q)...,)
                if sum(val.>=0) == C
                    mv.pd = 1
                    push!(multiVariable_convex, mv)
                end
                if sum(val.<=0) == C
                    mv.pd = -1
                    push!(multiVariable_convex, mv)
                end
                if sum(val.==0) == C
                    mv.pd = 0
                end
                if mv.pd == 0
                    alpha = Array{Float64}(undef,C)
                    lamda_min = minimum(val)
                    added = true
                    if sum(diag(Q).==0) == C
                        if length(mv.qVarsId) < 3
                            added = false
                        end
                    end
                    for k in 1:C
                        alpha[k] = max(0,  min(-lamda_min, sum(abs.(Q[k,:])) - 2*Q[k,k]))
                        varId =  mv.qVarsId[k]
                        if !haskey(qbVarsId, (varId, varId))
                            added = false
                            break
                        end
                    end
                    if added
                        mv.alpha = alpha
                        push!(multiVariable_aBB, mv)
                    end
                end
            end
	    push!(multiVariable_list, MultiVariableCon(mvs, copy(affcopy)))
        end
    end

    # create RLT
    EqVconstr = GenericAffExpr{Float64,VariableRef}[]
    
    linConstr1 = JuMP.all_constraints(P,GenericAffExpr{Float64,VariableRef},MathOptInterface.EqualTo{Float64})
    linConstr2 = JuMP.all_constraints(P,GenericAffExpr{Float64,VariableRef},MathOptInterface.GreaterThan{Float64})
    linConstr3 = JuMP.all_constraints(P,GenericAffExpr{Float64,VariableRef},MathOptInterface.LessThan{Float64})
    linConstr = [linConstr1;linConstr2;linConstr3]
    
    for i = 1:length(linConstr)
        con = JuMP.constraint_object(linConstr[i]).func
        constant=con.constant
        for varId in branchVarsId
            newcon = copy(con, m)
            aff = newcon
            accept = true
            affvars = [var for (var,coeff) in aff.terms]
            affcoeffs = [coeff for (var,coeff) in aff.terms]
            for k in 1:length(affvars)
                varIdinEq = affvars[k].index.value
                if haskey(qbVarsId, (varId, varIdinEq))
                    affvars[k] = VariableRef(m,m.ext[:idx_map][qbVarsId[varId, varIdinEq]])
                else
                    accept = false
                    break
                end
            end
            if accept
                if constant != 0
                   push!(aff.vars, VariableRef(m, m.ext[:idx_map][varId]))
                   push!(aff.coeffs, -constant)
                   newcon.lb = 0
                   newcon.ub = 0
                end
                push!(EqVconstr, newcon)
            end
        end
    end


    pr = PreprocessResult()
    pr.branchVarsId = branchVarsId
    pr.qbVarsId = qbVarsId
    pr.EqVconstr = EqVconstr
    pr.multiVariable_list = multiVariable_list
    pr.multiVariable_convex = multiVariable_convex
    pr.multiVariable_aBB = multiVariable_aBB
  
    pr.expVariable_list = expVariable_list
    pr.logVariable_list = logVariable_list
    pr.powerVariable_list = powerVariable_list
    pr.monomialVariable_list = monomialVariable_list 
    return pr, P
end

function findnode(c, nlinconstrs)
    nodeid = 1	 
    for i in 1:length(nlinconstrs)
	ncon = nlinconstrs[i]
	if c <= ncon  
	    nodeid = i
	    break
	end
    end
    return nodeid
end

function squeeze!(affEx::AffExpr)
    aff = affEx.terms
    vars = [var for (var,coeff) in aff]
    coeffs = [coeff for (var,coeff) in aff]
     redund = true
     while redund
         redund = false
         for k in 1:length(vars)
             x = vars[k]
             equal_index = findall(vars .== x)
             if length(equal_index) > 1
                 coeffs[k] = sum(coeffs[equal_index])
                 deleteat!(vars, equal_index[2:end])
                 deleteat!(coeffs, equal_index[2:end])
                 redund = true
                 break
             end
         end
     end
     affEx.terms = DataStructures.OrderedDict(zip(vars,coeffs))
end

function squeeze!(quadEx::QuadExpr)
    quad = quadEx.terms
    qvars1 = [qvar.a for (qvar,coeff) in quad]
    qvars2 = [qvar.b for (qvar,coeff) in quad]
    qcoeffs = [coeff for (qvar,coeff) in quad]
    aff = quadEx.aff
    redund = true
    while redund
         redund = false
         for k in 1:length(qvars1)
             x = qvars1[k]
             y = qvars2[k]

             equal_index = []
             coeff = 0
             for j in 1:length(qvars1)
                 if (qvars1[j] == x && qvars2[j] == y)|| (qvars1[j] == y && qvars2[j] == x)
                     push!(equal_index, j)
                     coeff += qcoeffs[j]
                 end
             end

             if length(equal_index) > 1
                 qcoeffs[k] = coeff
                 deleteat!(qvars1, equal_index[2:end])
                 deleteat!(qvars2, equal_index[2:end])
                 deleteat!(qcoeffs, equal_index[2:end])
                 redund = true
                 break
             end
         end
     end
     squeeze!(aff)
     pairs = [JuMP.UnorderedPair(qvar1,qvar2) for (qvar1,qvar2) in zip(qvars1,qvars2)]
     quadEx.terms = DataStructures.OrderedDict(zip(pairs,qcoeffs))
end

using Distributions, SparseArrays, LinearAlgebra

function preprocessSto!(P)
    scenarios = PlasmoOld.getchildren(P)	 
    # provide initial value if not defined
    vars = JuMP.all_variables(P)
    for (idx,var) in enumerate(vars)
        if isnothing(start_value(var))
           JuMP.set_start_value(var,0)
        end
    end

    # provide bounds if not defined
    for var in vars
        if !JuMP.has_lower_bound(var)
            JuMP.set_lower_bound(var,default_lower_bound_value)
        end
        if !JuMP.has_upper_bound(var)
             JuMP.set_upper_bound(var,default_upper_bound_value)
        end
    end
    
    for (idx,scenario) in enumerate(scenarios)
    scenVars = JuMP.all_variables(scenario)
    	for var in scenVars
            if !JuMP.has_lower_bound(var)
                JuMP.set_lower_bound(var,default_lower_bound_value)
            end
            if !JuMP.has_upper_bound(var)
                JuMP.set_upper_bound(var,default_upper_bound_value)
            end
    	end
	if debug
	    println("debug")
	end    
    end
    # For binary variables. Need to check what this is for.
    updateStoFirstBounds!(P)    


    nscen = length(scenarios)
    pr_children = []
    for (idx,scen) in enumerate(scenarios)
        pr, scenarios[idx] = preprocess!(scen)
        push!(pr_children, pr)
        linConstr1 = JuMP.all_constraints(P,GenericAffExpr{Float64,VariableRef},MathOptInterface.EqualTo{Float64})
        linConstr2 = JuMP.all_constraints(P,GenericAffExpr{Float64,VariableRef},MathOptInterface.LessThan{Float64})
        linConstr3 = JuMP.all_constraints(P,GenericAffExpr{Float64,VariableRef},MathOptInterface.GreaterThan{Float64})
        linConstr = [linConstr1;linConstr2;linConstr3]
        for constr in linConstr
            Pvars = Pcon.terms.vars
            for (j, Pvar) in enumerate(Pvars)
                if (Pvar.model == scen)
                     Pvars[j] = VariableRef(scenarios[idx], Pvar.index)
                end
            end
        end
	scen = -1
    end
    return pr_children

end


function Stopreprocess!(P)
    pr_children = []
    scenarios = PlasmoOld.getchildren(P)
    for (idx,scenario) in enumerate(scenarios)
    	pr = preprocess!(scenario)
	push!(pr_children, pr)
    end
    return pr_children
end
